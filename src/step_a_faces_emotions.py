from collections import Counter, deque
from tqdm import tqdm
import time # Importante para controle de framerate se necessário

from utils_step_a import (
    garantir_diretorio,
    FileVideoStream,           # <--- Nova Classe Otimizada
    ler_metadados_video,
    criar_video_writer,
    detectar_faces,
    extrair_bbox_e_confianca,
    calcular_area_e_ar,
    AutoajusteLimiar,
    passa_filtros_geometricos,
    passa_filtro_confianca,
    passa_persistencia,
    recortar_rosto,
    analisar_emocao,
    desenhar_anotacoes,
    escrever_resumo,
    carregar_banco_faces,
    reconhecer_identidade
)

def criar_config() -> dict:
    return {
        "VIDEO_ENTRADA": "data/input.mp4",
        "PASTA_FACES_CONHECIDAS": "data/known_faces",
        "VIDEO_SAIDA": "outputs/stepA_annotated.mp4",
        "RESUMO_SAIDA": "outputs/stepA_summary.txt",

        "FRAME_STEP": 3,

        # OTIMIZAÇÃO: Trocado de 'opencv' para 'mediapipe'
        # Se quiser usar YOLO, mude para "yolov8" (precisa pip install ultralytics)
        "DETECTOR_BACKEND": "mediapipe", 
        "ENFORCE_DETECTION": False,

        "DEBUG": True,
        "DEBUG_MAX_FRAMES": 10,
        "FRAMES_WARMUP_ANALISADOS": 150,
        "K_PERSISTENCIA": 2,
        "TAMANHO_GRID": 60,
        "PAD_RATIO": 0.15,
    }

def run_faces_emotions():
    cfg = criar_config()
    garantir_diretorio("outputs")

    # 1. Carregar Banco de Faces
    known_encodings, known_names = carregar_banco_faces(cfg["PASTA_FACES_CONHECIDAS"])

    # 2. Iniciar Leitura de Vídeo OTIMIZADA (Thread)
    print(f"🚀 Iniciando leitura otimizada (Threaded) do vídeo: {cfg['VIDEO_ENTRADA']}")
    cap_thread = FileVideoStream(cfg["VIDEO_ENTRADA"]).start()
    
    # Dá um tempo para o buffer encher
    time.sleep(1.0)

    # Lê metadados (usando o objeto otimizado)
    info = ler_metadados_video(cap_thread)
    
    # Writer
    escritor = criar_video_writer(cfg["VIDEO_SAIDA"], info["fps"], info["largura"], info["altura"])

    # Configurações de Filtro
    limiares = {
        "MIN_AREA_FACE": 40 * 40,
        "MAX_AREA_FACE": int(0.60 * info["area_frame"]),
        "MIN_AR": 0.30, "MAX_AR": 3.00, "MIN_CONFIANCA": 0.00
    }
    
    auto = AutoajusteLimiar(cfg["FRAMES_WARMUP_ANALISADOS"], info["area_frame"], cfg["DEBUG"])
    historico_ids = deque(maxlen=10)
    contador_emocoes = Counter()
    
    frames_analisados = 0
    total_faces = 0
    faces_ultimo_frame = []
    
    barra = tqdm(total=info["total_frames"], desc="Passo A (Otimizado)")
    indice_frame = 0
    debug_prints = 0

    # LOOP PRINCIPAL
    while cap_thread.more():
        # Leitura não bloqueante
        ret, frame = cap_thread.read()
        
        if not ret:
            # Se a fila esvaziou mas o vídeo não acabou, espera
            if not cap_thread.stopped:
                time.sleep(0.01)
                continue
            else:
                break # Fim do vídeo

        if frame is None:
            break

        analisou_este_frame = False

        # AMOSTRAGEM
        if indice_frame % cfg["FRAME_STEP"] == 0:
            try:
                # Detecção (Agora usando MediaPipe backend = +Rápido)
                faces_detectadas = detectar_faces(
                    frame_bgr=frame,
                    detector_backend=cfg["DETECTOR_BACKEND"],
                    enforce_detection=cfg["ENFORCE_DETECTION"]
                )

                if cfg["DEBUG"] and debug_prints < cfg["DEBUG_MAX_FRAMES"]:
                    print(f"[DEBUG] frame={indice_frame} | faces={len(faces_detectadas)} | backend={cfg['DETECTOR_BACKEND']}")
                    debug_prints += 1

                faces_validas = []

                for face_dict in faces_detectadas:
                    dados = extrair_bbox_e_confianca(face_dict)
                    x, y, w, h = dados["x"], dados["y"], dados["w"], dados["h"]

                    if w <= 0 or h <= 0: continue

                    area, ar = calcular_area_e_ar(w, h)

                    # Autoajuste
                    if not auto.limiares_definidos:
                        auto.adicionar_amostra(area, ar, dados["tem_confianca"], dados["confianca"])
                        if auto.pronto_para_definir():
                            limiares = auto.definir_limiares(limiares)

                    # Filtros
                    if not passa_filtros_geometricos(area, ar, limiares): continue
                    if not passa_filtro_confianca(dados["tem_confianca"], dados["confianca"], limiares): continue
                    if not passa_persistencia(historico_ids, x, y, cfg["TAMANHO_GRID"], cfg["K_PERSISTENCIA"]): continue

                    # Recorte
                    face_crop = recortar_rosto(frame, x, y, w, h, info["largura"], info["altura"], cfg["PAD_RATIO"])
                    
                    # Emoção
                    res_emocao = analisar_emocao(face_crop)
                    if not res_emocao: continue
                    
                    emocao = res_emocao.get("dominant_emotion", "unknown")
                    
                    # Identidade
                    nome = "Desconhecido"
                    if known_encodings:
                        nome = reconhecer_identidade(face_crop, known_encodings, known_names)

                    faces_validas.append({
                        "x": x, "y": y, "w": w, "h": h,
                        "emocao": emocao,
                        "nome": nome
                    })

                faces_ultimo_frame = faces_validas
                frames_analisados += 1
                analisou_este_frame = True

            except Exception as e:
                faces_ultimo_frame = [] # Reset em caso de erro para não travar visual
                if cfg["DEBUG"]: print(f"Erro frame {indice_frame}: {e}")

        # Desenho e Gravação
        desenhar_anotacoes(frame, faces_ultimo_frame, info["largura"], info["altura"])
        
        if analisou_este_frame:
            total_faces += len(faces_ultimo_frame)
            for f in faces_ultimo_frame:
                contador_emocoes[f["emocao"]] += 1

        escritor.write(frame)
        indice_frame += 1
        barra.update(1)

    # Limpeza
    barra.close()
    cap_thread.release() # Para a thread
    escritor.release()

    escrever_resumo(cfg["RESUMO_SAIDA"], {
        "video": cfg["VIDEO_ENTRADA"],
        "total_faces": total_faces,
        "contador_emocoes": contador_emocoes,
        "frames_totais": info["total_frames"],
        "frames_analisados": frames_analisados,
        "frame_step": cfg["FRAME_STEP"],
        "limiares": limiares,
        "k_persistencia": cfg["K_PERSISTENCIA"],
        "tamanho_grid": cfg["TAMANHO_GRID"]
    })

    print("✅ Passo A Otimizado Finalizado.")

if __name__ == "__main__":
    run_faces_emotions()
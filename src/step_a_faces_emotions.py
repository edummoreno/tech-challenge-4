from collections import Counter, deque
from tqdm import tqdm

from utils_step_a import (
    garantir_diretorio,
    abrir_video,
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
    carregar_banco_faces,      # <--- Importante
    reconhecer_identidade      # <--- Importante
)

# ============================================================
# CONFIG (didático)
# ============================================================
def criar_config() -> dict:
    return {
        # caminhos
        "VIDEO_ENTRADA": "data/input.mp4",
        "PASTA_FACES_CONHECIDAS": "data/known_faces", # <--- Config da pasta de fotos
        "VIDEO_SAIDA": "outputs/stepA_annotated.mp4",
        "RESUMO_SAIDA": "outputs/stepA_summary.txt",

        # amostragem temporal (performance)
        "FRAME_STEP": 3,

        # deepface / detecção
        "DETECTOR_BACKEND": "opencv",
        "ENFORCE_DETECTION": False,

        # debug
        "DEBUG": True,
        "DEBUG_MAX_FRAMES": 10,

        # patch 1 (warm-up)
        "FRAMES_WARMUP_ANALISADOS": 150,

        # patch 3 (persistência)
        "K_PERSISTENCIA": 2,
        "TAMANHO_GRID": 60,

        # emoção (crop)
        "PAD_RATIO": 0.15,   # 15% de margem no recorte do rosto
    }


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================
def run_faces_emotions():
    cfg = criar_config()

    # garante pasta outputs
    garantir_diretorio("outputs")

    # 1. CARREGAR CONHECIMENTOS (FOTOS)
    known_encodings, known_names = carregar_banco_faces(cfg["PASTA_FACES_CONHECIDAS"])

    # abre vídeo e lê metadados
    cap = abrir_video(cfg["VIDEO_ENTRADA"])
    info = ler_metadados_video(cap)

    fps = info["fps"]
    largura = info["largura"]
    altura = info["altura"]
    total_frames = info["total_frames"]
    area_frame = info["area_frame"]

    # cria writer do vídeo anotado
    escritor = criar_video_writer(cfg["VIDEO_SAIDA"], fps, largura, altura)

    # ============================================================
    # LIMIARES (fallback inicial “frouxo”)
    # ============================================================
    limiares = {
        "MIN_AREA_FACE": 40 * 40,
        "MAX_AREA_FACE": int(0.60 * area_frame),
        "MIN_AR": 0.30,
        "MAX_AR": 3.00,
        "MIN_CONFIANCA": 0.00
    }

    # Patch 1: autoajuste
    auto = AutoajusteLimiar(
        frames_warmup_analisados=cfg["FRAMES_WARMUP_ANALISADOS"],
        area_frame=area_frame,
        debug=cfg["DEBUG"]
    )

    # Patch 3: persistência
    historico_ids = deque(maxlen=10)

    # contadores para summary
    contador_emocoes = Counter()
    # (Opcional) contador de quem apareceu
    # contador_identidades = Counter() 

    frames_analisados = 0
    total_faces = 0

    # “memória” das faces do último frame analisado
    faces_ultimo_frame = []

    barra = tqdm(total=total_frames if total_frames > 0 else None, desc="Passo A — Faces + Emoções")
    debug_prints = 0
    indice_frame = 0

    # ============================================================
    # LOOP PRINCIPAL
    # ============================================================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        analisou_este_frame = False

        # ------------------------------------------------------------
        # AMOSTRAGEM TEMPORAL
        # ------------------------------------------------------------
        if indice_frame % cfg["FRAME_STEP"] == 0:
            try:
                faces_detectadas = detectar_faces(
                    frame_bgr=frame,
                    detector_backend=cfg["DETECTOR_BACKEND"],
                    enforce_detection=cfg["ENFORCE_DETECTION"]
                )

                if cfg["DEBUG"] and debug_prints < cfg["DEBUG_MAX_FRAMES"]:
                    print(f"[DEBUG] frame_idx={indice_frame} | faces_detectadas={len(faces_detectadas)}")
                    debug_prints += 1

                faces_validas = []

                # ------------------------------------------------------------
                # PROCESSA CADA FACE
                # ------------------------------------------------------------
                for face_dict in faces_detectadas:
                    dados = extrair_bbox_e_confianca(face_dict)
                    x, y, w, h = dados["x"], dados["y"], dados["w"], dados["h"]

                    # bbox inválida
                    if w <= 0 or h <= 0:
                        continue

                    area, ar = calcular_area_e_ar(w, h)

                    # PATCH 1 — AUTOAJUSTE
                    if not auto.limiares_definidos:
                        auto.adicionar_amostra(area, ar, dados["tem_confianca"], dados["confianca"])
                        if auto.pronto_para_definir():
                            limiares = auto.definir_limiares(limiares)

                    # PATCH 1 & 2 — FILTROS
                    if not passa_filtros_geometricos(area, ar, limiares):
                        continue
                    if not passa_filtro_confianca(dados["tem_confianca"], dados["confianca"], limiares):
                        continue

                    # PATCH 3 — PERSISTÊNCIA
                    if not passa_persistencia(historico_ids, x, y, cfg["TAMANHO_GRID"], cfg["K_PERSISTENCIA"]):
                        continue

                    # ====================================================
                    # 1. RECORTE E ANÁLISE DE EMOÇÃO
                    # ====================================================
                    face_crop = recortar_rosto(
                        frame_bgr=frame,
                        x=x, y=y, w=w, h=h,
                        largura=largura, altura=altura,
                        pad_ratio=cfg["PAD_RATIO"]
                    )

                    resultado = analisar_emocao(face_crop)
                    
                    # Se falhar a emoção, pula essa face
                    if resultado is None:
                        continue

                    # Extrai dados da emoção AGORA (antes de usar no append)
                    emocao = resultado.get("dominant_emotion", "unknown")
                    dist = resultado.get("emotion", {}) or {}
                    score = float(dist.get(emocao, 0.0)) if isinstance(dist, dict) else 0.0

                    # ====================================================
                    # 2. IDENTIFICAÇÃO (QUEM É?)
                    # ====================================================
                    nome_identificado = "Desconhecido"
                    if known_encodings:
                        nome_identificado = reconhecer_identidade(face_crop, known_encodings, known_names)

                    # ====================================================
                    # 3. SALVAR TUDO (UMA ÚNICA VEZ)
                    # ====================================================
                    faces_validas.append({
                        "x": x, "y": y, "w": w, "h": h,
                        "emocao": emocao,
                        "score": score,
                        "nome": nome_identificado
                    })

                faces_ultimo_frame = faces_validas
                frames_analisados += 1
                analisou_este_frame = True

            except Exception as e:
                # Anti “caixa fantasma”: se falhar, não reutiliza detecção antiga
                faces_ultimo_frame = []
                if cfg["DEBUG"]:
                    print(f"[DEBUG] Exception no frame {indice_frame}: {repr(e)}")

        # ------------------------------------------------------------
        # DESENHO NO VÍDEO (sempre)
        # ------------------------------------------------------------
        desenhar_anotacoes(frame, faces_ultimo_frame, largura, altura)

        # ------------------------------------------------------------
        # AGREGAÇÃO TEMPORAL (somente quando analisou)
        # ------------------------------------------------------------
        if analisou_este_frame:
            total_faces += len(faces_ultimo_frame)
            for f in faces_ultimo_frame:
                contador_emocoes[f["emocao"]] += 1

        escritor.write(frame)
        indice_frame += 1
        barra.update(1)

    # ============================================================
    # FINALIZAÇÃO
    # ============================================================
    barra.close()
    cap.release()
    escritor.release()

    escrever_resumo(cfg["RESUMO_SAIDA"], {
        "video": cfg["VIDEO_ENTRADA"],
        "frames_totais": total_frames,
        "frames_analisados": frames_analisados,
        "frame_step": cfg["FRAME_STEP"],
        "total_faces": total_faces,
        "limiares": limiares,
        "k_persistencia": cfg["K_PERSISTENCIA"],
        "tamanho_grid": cfg["TAMANHO_GRID"],
        "contador_emocoes": contador_emocoes,
    })

    print("✅ Passo A finalizado com sucesso.")
    print(f"🎥 Vídeo: {cfg['VIDEO_SAIDA']}")
    print(f"📝 Resumo: {cfg['RESUMO_SAIDA']}")


if __name__ == "__main__":
    run_faces_emotions()
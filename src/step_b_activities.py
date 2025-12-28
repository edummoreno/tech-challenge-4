import cv2
import os
from collections import Counter
from tqdm import tqdm

# Importa utilitários do Step A (para abrir vídeo/writer)
from utils_step_a import abrir_video, ler_metadados_video, criar_video_writer, garantir_diretorio, escrever_resumo

# Importa utilitários do Step B (lógica das poses)
from utils_step_b import iniciar_pose, classificar_atividade, desenhar_esqueleto, desenhar_atividade

# ============================================================
# CONFIG STEP B
# ============================================================
def criar_config_b():
    return {
        # IMPORTANTE: A entrada do B é a saída do A (encadeamento)
        "VIDEO_ENTRADA": "outputs/stepA_annotated.mp4", 
        "VIDEO_SAIDA": "outputs/stepB_final.mp4",
        "RESUMO_SAIDA": "outputs/stepB_summary.txt",
        "FRAME_STEP": 1 
    }

# ============================================================
# FUNÇÃO PRINCIPAL (QUE ESTAVA FALTANDO)
# ============================================================
def run_activities():
    cfg = criar_config_b()
    
    if not os.path.exists(cfg["VIDEO_ENTRADA"]):
        print(f"❌ Erro: Saída do Step A não encontrada ({cfg['VIDEO_ENTRADA']}). Rode o Step A primeiro.")
        return

    garantir_diretorio("outputs")

    # Inicializa Pose Detector
    pose_detector = iniciar_pose()

    # Abre vídeo
    cap = abrir_video(cfg["VIDEO_ENTRADA"])
    info = ler_metadados_video(cap)
    
    # Prepara gravador de vídeo
    writer = criar_video_writer(cfg["VIDEO_SAIDA"], info["fps"], info["largura"], info["altura"])

    # Contadores
    contador_atividades = Counter()
    total_frames = info["total_frames"]
    
    barra = tqdm(total=total_frames, desc="Passo B — Detecção de Atividades")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Converte para RGB (MediaPipe usa RGB, OpenCV usa BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processa Pose
        results = pose_detector.process(frame_rgb)
        
        atividade_atual = "Nenhuma pessoa detectada"
        
        if results.pose_landmarks:
            # 1. Desenha esqueleto
            desenhar_esqueleto(frame, results)
            
            # 2. Classifica atividade
            atividade_atual = classificar_atividade(results.pose_landmarks.landmark)
            contador_atividades[atividade_atual] += 1
        
        # 3. Escreve no vídeo
        desenhar_atividade(frame, atividade_atual)

        # Salva frame
        writer.write(frame)
        barra.update(1)

    # Finaliza
    cap.release()
    writer.release()
    pose_detector.close()
    barra.close()

    # Gera resumo do Step B
    escrever_resumo_b(cfg["RESUMO_SAIDA"], contador_atividades, info)

    print("✅ Passo B finalizado com sucesso.")
    print(f"🎥 Vídeo Final: {cfg['VIDEO_SAIDA']}")

def escrever_resumo_b(filepath, contador, info):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=== PASSO B — Detecção de Atividades ===\n")
        f.write(f"Total Frames: {info['total_frames']}\n")
        f.write("Contagem de Atividades:\n")
        for atv, count in contador.most_common():
            f.write(f"- {atv}: {count} frames\n")

if __name__ == "__main__":
    run_activities()
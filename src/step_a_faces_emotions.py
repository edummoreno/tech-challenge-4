import os
import cv2
import numpy as np
from tqdm import tqdm
from collections import Counter, deque
from deepface import DeepFace

# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================
VIDEO_ENTRADA = os.path.join("data", "input.mp4")
VIDEO_SAIDA = os.path.join("outputs", "stepA_annotated.mp4")
RESUMO_SAIDA = os.path.join("outputs", "stepA_summary.txt")

# AMOSTRAGEM TEMPORAL (performance)
# - 3 costuma ser um bom equilíbrio; 2 é mais detalhado e mais lento
FRAME_STEP = 3

# Detector facial:
# - "opencv" = rápido, mas pode ser menos estável e pode não fornecer confidence
# - "retinaface" = mais preciso (recomendado se tiver muitos falsos positivos), porém mais lento
DETECTOR_BACKEND = "opencv"

# Se True, pode dar erro quando não encontra faces; se False, fica resiliente
ENFORCE_DETECTION = False

# DEBUG (imprime infos úteis no console)
DEBUG = True
DEBUG_MAX_FRAMES = 10

# ============================================================
# PATCH 1 — AUTOAJUSTE (WARM-UP) POR ESTATÍSTICA ROBUSTA
# ============================================================
# Ideia:
# - Coletar amostras (área, aspect ratio e confiança) nos primeiros frames analisados
# - Definir limiares automaticamente usando percentis (robusto a outliers)
FRAMES_WARMUP_ANALISADOS = 150
amostras_area = []
amostras_ar = []
amostras_confianca = []
limiares_auto_definidos = False

# ============================================================
# PATCH 3 — PERSISTÊNCIA TEMPORAL (ANTI-RUÍDO)
# ============================================================
# Ideia:
# - Falso positivo aparece 1 frame e some; face real tende a persistir
K_PERSISTENCIA = 2

# Quantização para criar um id aproximado da face por posição
# (quanto maior, mais tolerante a "jitter" na bbox)
TAMANHO_GRID = 60

# Histórico recente de ids de faces
historico_faces = deque(maxlen=10)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
def garantir_diretorio(caminho: str):
    os.makedirs(caminho, exist_ok=True)


def normalizar_resultado_analise(resultado):
    """
    DeepFace.analyze às vezes retorna dict e às vezes list[dict].
    Normaliza para dict.
    """
    if resultado is None:
        return None
    if isinstance(resultado, list):
        return resultado[0] if len(resultado) > 0 else None
    if isinstance(resultado, dict):
        return resultado
    return None


# ============================================================
# MAIN
# ============================================================
def main():
    global limiares_auto_definidos
    global MIN_AREA_FACE, MAX_AREA_FACE, MIN_AR, MAX_AR, MIN_CONFIANCA

    if not os.path.exists(VIDEO_ENTRADA):
        raise FileNotFoundError(f"Vídeo não encontrado: {VIDEO_ENTRADA}")

    garantir_diretorio("outputs")

    cap = cv2.VideoCapture(VIDEO_ENTRADA)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir o vídeo")

    # Metadados do vídeo
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    area_frame = largura * altura

    # ============================================================
    # FALLBACK INICIAL (bem frouxo)
    # ============================================================
    # Por quê?
    # - Evita "nascer travado" antes do warm-up aprender limiares do vídeo.
    MIN_AREA_FACE = 40 * 40
    MAX_AREA_FACE = int(0.60 * area_frame)
    MIN_AR = 0.30
    MAX_AR = 3.00
    MIN_CONFIANCA = 0.00  # importante se o backend não fornece confidence

    # Escritor do vídeo (sem áudio por limitação do OpenCV VideoWriter)
    escritor = cv2.VideoWriter(
        VIDEO_SAIDA,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (largura, altura)
    )

    contador_emocoes = Counter()
    frames_analisados = 0
    total_faces_detectadas = 0
    faces_ultimo_frame = []

    barra = tqdm(total=total_frames if total_frames > 0 else None, desc="Passo A — Faces + Emoções")
    indice_frame = 0
    debug_frames_printados = 0

    # ============================================================
    # LOOP PRINCIPAL
    # ============================================================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        analisou_este_frame = False

        # ====================================================
        # AMOSTRAGEM TEMPORAL (performance)
        # ====================================================
        if indice_frame % FRAME_STEP == 0:
            try:
                # ====================================================
                # DETECÇÃO FACIAL + (POSSÍVEL) CONFIANÇA
                # ====================================================
                faces_detectadas = DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=ENFORCE_DETECTION,
                    align=True
                )

                if DEBUG and debug_frames_printados < DEBUG_MAX_FRAMES:
                    print(f"[DEBUG] frame_idx={indice_frame} | faces_detectadas={len(faces_detectadas)}")
                    debug_frames_printados += 1

                faces_validas = []

                for face in faces_detectadas:
                    # Confidence pode vir None (depende do backend/versão)
                    confianca = face.get("confidence", None)
                    tem_confianca = (confianca is not None)

                    # Se não há confidence, não queremos matar tudo.
                    # (o filtro por confiança só será aplicado se tem_confianca=True)
                    if not tem_confianca:
                        confianca = 1.0

                    # Bounding box do detector
                    area_face = face.get("facial_area", {})
                    x = int(area_face.get("x", 0))
                    y = int(area_face.get("y", 0))
                    w = int(area_face.get("w", 0))
                    h = int(area_face.get("h", 0))

                    if w <= 0 or h <= 0:
                        continue

                    # Métricas geométricas
                    area = w * h
                    ar = w / float(h)

                    # ====================================================
                    # PATCH 1 — AUTOAJUSTE (Fase A: COLETA / warm-up)
                    # ====================================================
                    if not limiares_auto_definidos:
                        # Evita coletar "absurdos" (ex: tela inteira)
                        if area < 0.6 * area_frame:
                            amostras_area.append(area)
                            amostras_ar.append(ar)

                            # Só coletamos confiança se ela existir de verdade
                            if tem_confianca:
                                amostras_confianca.append(float(confianca))

                        # Quando tiver amostras suficientes, calcula percentis
                        if len(amostras_area) >= FRAMES_WARMUP_ANALISADOS:
                            MIN_AREA_FACE = int(np.percentile(amostras_area, 10))
                            MAX_AREA_FACE = int(np.percentile(amostras_area, 95))
                            MIN_AR = float(np.percentile(amostras_ar, 5))
                            MAX_AR = float(np.percentile(amostras_ar, 95))

                            # Guardrail: evita colapso do AR (ex.: MIN_AR=MAX_AR=1.0)
                            # Isso pode acontecer se o detector retornar bbox quase sempre quadrada.
                            if (MAX_AR - MIN_AR) < 0.15:
                                MIN_AR = 0.6
                                MAX_AR = 1.6

                            if len(amostras_confianca) > 0:
                                MIN_CONFIANCA = float(np.percentile(amostras_confianca, 20))
                            else:
                                MIN_CONFIANCA = 0.0

                            limiares_auto_definidos = True

                            if DEBUG:
                                print("[DEBUG] Autoajuste definido!")
                                print(f"        MIN_AREA_FACE={MIN_AREA_FACE} | MAX_AREA_FACE={MAX_AREA_FACE}")
                                print(f"        MIN_AR={MIN_AR:.3f} | MAX_AR={MAX_AR:.3f}")
                                print(f"        MIN_CONFIANCA={MIN_CONFIANCA:.3f} (amostras_conf={len(amostras_confianca)})")

                    # ====================================================
                    # PATCH 1 — AUTOAJUSTE (Fase B: APLICAÇÃO)
                    # ====================================================
                    if area < MIN_AREA_FACE or area > MAX_AREA_FACE:
                        continue
                    if not (MIN_AR <= ar <= MAX_AR):
                        continue

                    # ====================================================
                    # PATCH 2 — FILTRO POR CONFIANÇA (se existir)
                    # ====================================================
                    if tem_confianca and float(confianca) < MIN_CONFIANCA:
                        continue

                    # ====================================================
                    # PATCH 3 — PERSISTÊNCIA TEMPORAL
                    # ====================================================
                    id_face = (round(x / TAMANHO_GRID), round(y / TAMANHO_GRID))
                    historico_faces.append(id_face)
                    if K_PERSISTENCIA > 1 and historico_faces.count(id_face) < K_PERSISTENCIA:
                        continue

                    # ====================================================
                    # EMOÇÃO — CORREÇÃO IMPORTANTE (para não "travar" em angry)
                    # ====================================================
                    # Em vez de usar face.get("face") (pode vir RGB/float/normalizado),
                    # recortamos diretamente do frame original (BGR uint8).
                    pad = int(0.15 * max(w, h))  # margem ajuda a pegar o rosto inteiro

                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(largura, x + w + pad)
                    y2 = min(altura, y + h + pad)

                    face_crop = frame[y1:y2, x1:x2]  # BGR uint8
                    if face_crop.size == 0:
                        continue

                    # Se a face for muito pequena, redimensiona (evita inputs minúsculos)
                    if face_crop.shape[0] < 48 or face_crop.shape[1] < 48:
                        face_crop = cv2.resize(face_crop, (96, 96), interpolation=cv2.INTER_LINEAR)

                    # Tenta rodar emoção sem detecção (quando suportado)
                    try:
                        resultado_emocao = DeepFace.analyze(
                            img_path=face_crop,
                            actions=["emotion"],
                            enforce_detection=False,
                            detector_backend="skip"
                        )
                    except Exception:
                        resultado_emocao = DeepFace.analyze(
                            img_path=face_crop,
                            actions=["emotion"],
                            enforce_detection=False
                        )

                    resultado_emocao = normalizar_resultado_analise(resultado_emocao)
                    if resultado_emocao is None:
                        continue

                    emocao = resultado_emocao.get("dominant_emotion", "unknown")
                    dist = resultado_emocao.get("emotion", {}) or {}
                    score = float(dist.get(emocao, 0.0)) if isinstance(dist, dict) else 0.0

                    # DEBUG opcional: ver top2 emoções para garantir que não está "travado"
                    if DEBUG and isinstance(dist, dict) and debug_frames_printados <= DEBUG_MAX_FRAMES:
                        top2 = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:2]
                        # Mostra só se realmente tiver conteúdo
                        if len(top2) == 2:
                            print(f"[DEBUG] top2 emoções: {top2}")

                    faces_validas.append({
                        "x": x, "y": y, "w": w, "h": h,
                        "emocao": emocao,
                        "score": score,
                        "confianca": float(confianca) if confianca is not None else -1.0
                    })

                faces_ultimo_frame = faces_validas
                frames_analisados += 1
                analisou_este_frame = True

            except Exception as e:
                faces_ultimo_frame = []
                if DEBUG:
                    print(f"[DEBUG] Exception no frame {indice_frame}: {repr(e)}")

        # ====================================================
        # DESENHO NO VÍDEO
        # ====================================================
        for f in faces_ultimo_frame:
            x, y, w, h = f["x"], f["y"], f["w"], f["h"]
            emocao, score = f["emocao"], f["score"]

            # clamp para evitar sair do frame
            x = max(0, min(x, largura - 1))
            y = max(0, min(y, altura - 1))
            w = max(0, min(w, largura - x))
            h = max(0, min(h, altura - y))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            texto = f"{emocao} ({score:.0f}%)"
            cv2.putText(
                frame, texto, (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        # ====================================================
        # AGREGAÇÃO TEMPORAL (perfil emocional ao longo do vídeo)
        # ====================================================
        if analisou_este_frame:
            total_faces_detectadas += len(faces_ultimo_frame)
            for f in faces_ultimo_frame:
                contador_emocoes[f["emocao"]] += 1

        escritor.write(frame)
        indice_frame += 1
        barra.update(1)

    barra.close()
    cap.release()
    escritor.release()

    # ============================================================
    # RESUMO TEXTUAL
    # ============================================================
    with open(RESUMO_SAIDA, "w", encoding="utf-8") as f:
        f.write("=== PASSO A — Reconhecimento Facial + Emoções ===\n")
        f.write(f"Vídeo: {VIDEO_ENTRADA}\n")
        f.write(f"Frames totais: {total_frames}\n")
        f.write(f"Frames analisados (amostragem, step={FRAME_STEP}): {frames_analisados}\n")
        f.write(f"Total de faces detectadas (após filtros): {total_faces_detectadas}\n")
        f.write("Limiares finais:\n")
        f.write(f"  MIN_AREA_FACE={MIN_AREA_FACE}\n")
        f.write(f"  MAX_AREA_FACE={MAX_AREA_FACE}\n")
        f.write(f"  MIN_AR={MIN_AR:.3f}\n")
        f.write(f"  MAX_AR={MAX_AR:.3f}\n")
        f.write(f"  MIN_CONFIANCA={MIN_CONFIANCA:.3f}\n")
        f.write(f"  K_PERSISTENCIA={K_PERSISTENCIA}\n")
        f.write(f"  TAMANHO_GRID={TAMANHO_GRID}\n\n")

        f.write("Top emoções:\n")
        for emocao, qtd in contador_emocoes.most_common(10):
            f.write(f"- {emocao}: {qtd}\n")

    print("✅ Passo A finalizado com sucesso.")
    print(f"🎥 Vídeo: {VIDEO_SAIDA}")
    print(f"📝 Resumo: {RESUMO_SAIDA}")

    # Dica rápida:
    # - Se ainda estiver “travando” em uma emoção só, teste DETECTOR_BACKEND="retinaface"
    # - Depois que estiver ok, pode voltar FRAME_STEP=3 para rodar mais rápido


if __name__ == "__main__":
    main()

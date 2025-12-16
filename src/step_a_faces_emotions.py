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
# - FRAME_STEP = 3 costuma ser um bom equilíbrio;
# - FRAME_STEP = 2 pega mais variações (melhor “temporal”), porém fica mais lento.
FRAME_STEP = 3

# Detector facial:
# - "opencv" = rápido, mas pode ser menos estável e pode não fornecer confidence
# - "retinaface" = mais preciso (menos falso positivo), porém mais lento
DETECTOR_BACKEND = "opencv"
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
# - Falso positivo aparece em 1 frame e some.
# - Face real tende a persistir por mais frames.
K_PERSISTENCIA = 2

# Quantização para criar um id aproximado da face por posição
# (quanto maior, mais tolerante a “jitter” na bbox)
TAMANHO_GRID = 60

# Histórico recente de ids de faces (para persistência)
historico_faces = deque(maxlen=10)


# ============================================================
# FUNÇÕES AUXILIARES (simples e diretas)
# ============================================================
def garantir_diretorio(caminho: str):
    """Cria diretório se não existir."""
    os.makedirs(caminho, exist_ok=True)


def normalizar_resultado_analise(resultado):
    """
    DeepFace.analyze às vezes retorna dict e às vezes list[dict].
    Normaliza sempre para dict.
    """
    if resultado is None:
        return None
    if isinstance(resultado, list):
        return resultado[0] if len(resultado) > 0 else None
    if isinstance(resultado, dict):
        return resultado
    return None


def abrir_video(caminho_video: str):
    """Abre o vídeo e valida."""
    if not os.path.exists(caminho_video):
        raise FileNotFoundError(f"Vídeo não encontrado: {caminho_video}")

    cap = cv2.VideoCapture(caminho_video)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir o vídeo")
    return cap


def ler_metadados_video(cap):
    """Lê fps, largura, altura e total de frames."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    return fps, largura, altura, total_frames


def criar_video_writer(caminho_saida: str, fps: float, largura: int, altura: int):
    """Cria o writer do vídeo de saída (OpenCV não escreve áudio)."""
    return cv2.VideoWriter(
        caminho_saida,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (largura, altura),
    )


def detectar_faces(frame):
    """
    DETECÇÃO FACIAL:
    Usamos DeepFace.extract_faces porque ele fornece:
    - bounding box (facial_area)
    - face crop (mas NÃO vamos confiar nele para emoção)
    - confidence (às vezes, depende do backend)
    """
    return DeepFace.extract_faces(
        img_path=frame,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=ENFORCE_DETECTION,
        align=True
    )


def extrair_bbox_e_confianca(face_dict):
    """
    Extrai:
    - confianca (se existir)
    - x, y, w, h do facial_area
    """
    confianca = face_dict.get("confidence", None)
    tem_confianca = (confianca is not None)

    area_face = face_dict.get("facial_area", {}) or {}
    x = int(area_face.get("x", 0))
    y = int(area_face.get("y", 0))
    w = int(area_face.get("w", 0))
    h = int(area_face.get("h", 0))

    return tem_confianca, confianca, x, y, w, h


def calcular_area_e_ar(w: int, h: int):
    """Calcula área e aspect ratio (AR = w/h)."""
    area = w * h
    ar = w / float(h) if h != 0 else 0.0
    return area, ar


def coletar_amostras_warmup(area, ar, tem_confianca, confianca, area_frame):
    """
    PATCH 1 (Fase A):
    Coleta estatísticas do vídeo real (warm-up) para autoajuste.
    """
    # Evita coletar outliers absurdos (ex: bbox “tela inteira”)
    if area < 0.6 * area_frame:
        amostras_area.append(area)
        amostras_ar.append(ar)

        # Só coletamos confiança se ela existir de verdade
        if tem_confianca:
            amostras_confianca.append(float(confianca))


def definir_limiares_por_percentis():
    """
    PATCH 1 (Fase A → B):
    Define limiares adaptativos a partir dos percentis das amostras.
    """
    global limiares_auto_definidos
    global MIN_AREA_FACE, MAX_AREA_FACE, MIN_AR, MAX_AR, MIN_CONFIANCA

    MIN_AREA_FACE = int(np.percentile(amostras_area, 10))
    MAX_AREA_FACE = int(np.percentile(amostras_area, 95))
    MIN_AR = float(np.percentile(amostras_ar, 5))
    MAX_AR = float(np.percentile(amostras_ar, 95))

    # Guardrail: evita colapso do AR (ex.: MIN_AR=MAX_AR=1.0)
    if (MAX_AR - MIN_AR) < 0.15:
        MIN_AR = 0.6
        MAX_AR = 1.6

    # Confiança: só se houver amostras (depende do backend)
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


def passa_filtros_geometricos(area, ar):
    """
    PATCH 1 (Fase B):
    Aplica limiares geométricos (área e AR).
    """
    if area < MIN_AREA_FACE or area > MAX_AREA_FACE:
        return False
    if not (MIN_AR <= ar <= MAX_AR):
        return False
    return True


def passa_filtro_confianca(tem_confianca, confianca):
    """
    PATCH 2:
    Aplica filtro de confiança somente quando o backend fornece confidence.
    """
    if not tem_confianca:
        return True  # se não existe confidence, não bloqueia
    return float(confianca) >= MIN_CONFIANCA


def passa_persistencia(x, y):
    """
    PATCH 3:
    Exige persistência temporal usando um id aproximado por posição.
    """
    id_face = (round(x / TAMANHO_GRID), round(y / TAMANHO_GRID))
    historico_faces.append(id_face)

    if K_PERSISTENCIA <= 1:
        return True

    return historico_faces.count(id_face) >= K_PERSISTENCIA


def recortar_rosto_do_frame(frame, x, y, w, h, largura, altura, pad_ratio=0.15):
    """
    EMOÇÃO (correção importante):
    Para não “travar” em uma emoção (ex: angry), NÃO usamos face.get("face").
    Em vez disso, recortamos diretamente do frame original (BGR uint8).
    """
    pad = int(pad_ratio * max(w, h))

    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(largura, x + w + pad)
    y2 = min(altura, y + h + pad)

    face_crop = frame[y1:y2, x1:x2]
    return face_crop


def analisar_emocao(face_crop):
    """
    Roda DeepFace.analyze para emoção.
    Tentamos detector_backend='skip' (quando suportado) para evitar redetecção.
    """
    if face_crop is None or face_crop.size == 0:
        return None

    # Se o crop for muito pequeno, redimensiona para estabilizar o modelo
    if face_crop.shape[0] < 48 or face_crop.shape[1] < 48:
        face_crop = cv2.resize(face_crop, (96, 96), interpolation=cv2.INTER_LINEAR)

    try:
        resultado = DeepFace.analyze(
            img_path=face_crop,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="skip"
        )
    except Exception:
        resultado = DeepFace.analyze(
            img_path=face_crop,
            actions=["emotion"],
            enforce_detection=False
        )

    return normalizar_resultado_analise(resultado)


def desenhar_anotacoes(frame, faces, largura, altura):
    """
    Desenha bbox e texto no frame.
    """
    for f in faces:
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


def escrever_resumo(total_frames, frames_analisados, total_faces_detectadas):
    """
    Gera o summary.txt com estatísticas e limiares finais.
    """
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


# ============================================================
# MAIN
# ============================================================
def main():
    global MIN_AREA_FACE, MAX_AREA_FACE, MIN_AR, MAX_AR, MIN_CONFIANCA
    global limiares_auto_definidos

    garantir_diretorio("outputs")

    cap = abrir_video(VIDEO_ENTRADA)
    fps, largura, altura, total_frames = ler_metadados_video(cap)
    area_frame = largura * altura

    # ============================================================
    # FALLBACK INICIAL (bem frouxo)
    # ============================================================
    # Evita “nascer travado” antes do warm-up aprender limiares do vídeo.
    MIN_AREA_FACE = 40 * 40
    MAX_AREA_FACE = int(0.60 * area_frame)
    MIN_AR = 0.30
    MAX_AR = 3.00
    MIN_CONFIANCA = 0.00  # importante se backend não fornece confidence

    escritor = criar_video_writer(VIDEO_SAIDA, fps, largura, altura)

    # Contadores (para resumo)
    global contador_emocoes
    contador_emocoes = Counter()
    frames_analisados = 0
    total_faces_detectadas = 0

    # Mantém as faces do último frame analisado (pra desenhar em todos os frames)
    faces_ultimo_frame = []

    barra = tqdm(total=total_frames if total_frames > 0 else None, desc="Passo A — Faces + Emoções")
    debug_frames_printados = 0
    indice_frame = 0

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
                faces_detectadas = detectar_faces(frame)

                if DEBUG and debug_frames_printados < DEBUG_MAX_FRAMES:
                    print(f"[DEBUG] frame_idx={indice_frame} | faces_detectadas={len(faces_detectadas)}")
                    debug_frames_printados += 1

                faces_validas = []

                for face_dict in faces_detectadas:
                    tem_confianca, confianca, x, y, w, h = extrair_bbox_e_confianca(face_dict)

                    if w <= 0 or h <= 0:
                        continue

                    area, ar = calcular_area_e_ar(w, h)

                    # ====================================================
                    # PATCH 1 — AUTOAJUSTE (Fase A: COLETA / warm-up)
                    # ====================================================
                    if not limiares_auto_definidos:
                        coletar_amostras_warmup(area, ar, tem_confianca, confianca, area_frame)

                        # Quando tiver amostras suficientes, calcula percentis e fecha o autoajuste
                        if len(amostras_area) >= FRAMES_WARMUP_ANALISADOS:
                            definir_limiares_por_percentis()

                    # ====================================================
                    # PATCH 1 — AUTOAJUSTE (Fase B: APLICAÇÃO)
                    # ====================================================
                    if not passa_filtros_geometricos(area, ar):
                        continue

                    # ====================================================
                    # PATCH 2 — FILTRO POR CONFIANÇA (se existir)
                    # ====================================================
                    if not passa_filtro_confianca(tem_confianca, confianca):
                        continue

                    # ====================================================
                    # PATCH 3 — PERSISTÊNCIA TEMPORAL
                    # ====================================================
                    if not passa_persistencia(x, y):
                        continue

                    # ====================================================
                    # EMOÇÃO — recorte do frame original (corrige viés “angry”)
                    # ====================================================
                    face_crop = recortar_rosto_do_frame(frame, x, y, w, h, largura, altura, pad_ratio=0.15)
                    resultado_emocao = analisar_emocao(face_crop)
                    if resultado_emocao is None:
                        continue

                    emocao = resultado_emocao.get("dominant_emotion", "unknown")
                    dist = resultado_emocao.get("emotion", {}) or {}
                    score = float(dist.get(emocao, 0.0)) if isinstance(dist, dict) else 0.0

                    # DEBUG opcional: ver top2 emoções para garantir que não está “travado”
                    if DEBUG and isinstance(dist, dict) and debug_frames_printados <= DEBUG_MAX_FRAMES:
                        top2 = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:2]
                        if len(top2) == 2:
                            print(f"[DEBUG] top2 emoções: {top2}")

                    faces_validas.append({
                        "x": x, "y": y, "w": w, "h": h,
                        "emocao": emocao,
                        "score": score,
                        "confianca": float(confianca) if tem_confianca else -1.0
                    })

                faces_ultimo_frame = faces_validas
                frames_analisados += 1
                analisou_este_frame = True

            except Exception as e:
                # Anti “caixa fantasma”: se falhar, não reutiliza detecção antiga
                faces_ultimo_frame = []
                if DEBUG:
                    print(f"[DEBUG] Exception no frame {indice_frame}: {repr(e)}")

        # ====================================================
        # DESENHO NO VÍDEO (em todos os frames)
        # ====================================================
        desenhar_anotacoes(frame, faces_ultimo_frame, largura, altura)

        # ====================================================
        # AGREGAÇÃO TEMPORAL (perfil emocional)
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

    escrever_resumo(total_frames, frames_analisados, total_faces_detectadas)

    print("✅ Passo A finalizado com sucesso.")
    print(f"🎥 Vídeo: {VIDEO_SAIDA}")
    print(f"📝 Resumo: {RESUMO_SAIDA}")

    # Dica rápida:
    # - Se tiver muitos falsos positivos: DETECTOR_BACKEND="retinaface"
    # - Se quiser mais rápido: FRAME_STEP=3 (ou 4)
    # - Se quiser mais estável: aumente TAMANHO_GRID ou K_PERSISTENCIA


if __name__ == "__main__":
    main()

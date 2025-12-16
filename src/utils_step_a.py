import os
import cv2
import numpy as np
from deepface import DeepFace


# ============================================================
# UTILITÁRIOS DE ARQUIVO / VÍDEO
# ============================================================
def garantir_diretorio(caminho: str) -> None:
    """Cria diretório se não existir."""
    os.makedirs(caminho, exist_ok=True)


def abrir_video(caminho: str) -> cv2.VideoCapture:
    """Abre o vídeo e valida."""
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Vídeo não encontrado: {caminho}")

    cap = cv2.VideoCapture(caminho)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir o vídeo")
    return cap


def ler_metadados_video(cap: cv2.VideoCapture) -> dict:
    """Lê FPS, largura, altura, total_frames e área do frame."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    area_frame = largura * altura

    return {
        "fps": fps,
        "largura": largura,
        "altura": altura,
        "total_frames": total_frames,
        "area_frame": area_frame,
    }


def criar_video_writer(caminho_saida: str, fps: float, largura: int, altura: int) -> cv2.VideoWriter:
    """Cria o writer do vídeo de saída (OpenCV não escreve áudio)."""
    return cv2.VideoWriter(
        caminho_saida,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (largura, altura),
    )


# ============================================================
# DETECÇÃO DE FACES (DeepFace)
# ============================================================
def detectar_faces(frame_bgr, detector_backend: str, enforce_detection: bool):
    """
    Usa DeepFace.extract_faces para obter:
    - facial_area (x,y,w,h)
    - confidence (às vezes, depende do backend)
    Obs: o face_crop retornado pelo DeepFace pode vir normalizado.
    """
    return DeepFace.extract_faces(
        img_path=frame_bgr,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=True
    )


def extrair_bbox_e_confianca(face_dict: dict) -> dict:
    """
    Extrai bounding box e confiança do dict retornado pelo DeepFace.extract_faces.
    Retorna um dicionário padronizado.
    """
    confianca = face_dict.get("confidence", None)
    tem_confianca = (confianca is not None)

    facial_area = face_dict.get("facial_area", {}) or {}
    x = int(facial_area.get("x", 0))
    y = int(facial_area.get("y", 0))
    w = int(facial_area.get("w", 0))
    h = int(facial_area.get("h", 0))

    return {
        "x": x, "y": y, "w": w, "h": h,
        "tem_confianca": tem_confianca,
        "confianca": float(confianca) if tem_confianca else None,
    }


def calcular_area_e_ar(w: int, h: int) -> tuple:
    """
    Calcula:
    - area = w*h
    - ar (aspect ratio) = w/h
    """
    if w <= 0 or h <= 0:
        return 0, 0.0
    return w * h, (w / float(h))


# ============================================================
# PATCH 1 — AUTOAJUSTE (WARM-UP) POR PERCENTIS
# ============================================================
class AutoajusteLimiar:
    """
    Guarda amostras durante warm-up e depois calcula limiares por percentis.
    Isso torna o pipeline menos dependente de tuning manual por vídeo.
    """

    def __init__(self, frames_warmup_analisados: int, area_frame: int, debug: bool = False):
        self.frames_warmup_analisados = frames_warmup_analisados
        self.area_frame = area_frame
        self.debug = debug

        self.amostras_area = []
        self.amostras_ar = []
        self.amostras_confianca = []

        self.limiares_definidos = False

    def adicionar_amostra(self, area: int, ar: float, tem_confianca: bool, confianca: float | None):
        """
        Coleta amostras “realistas”.
        Evitamos outliers absurdos (ex.: bbox quase do tamanho do frame inteiro).
        """
        if area <= 0:
            return

        # evita coletar absurdos
        if area < 0.6 * self.area_frame:
            self.amostras_area.append(area)
            self.amostras_ar.append(ar)

            # confiança só se existir de verdade
            if tem_confianca and confianca is not None:
                self.amostras_confianca.append(float(confianca))

    def pronto_para_definir(self) -> bool:
        return (not self.limiares_definidos) and (len(self.amostras_area) >= self.frames_warmup_analisados)

    def definir_limiares(self, limiares: dict) -> dict:
        """
        Calcula percentis e atualiza o dict de limiares.
        Inclui guardrail para evitar colapso do AR (MIN_AR ~ MAX_AR).
        """
        limiares["MIN_AREA_FACE"] = int(np.percentile(self.amostras_area, 10))
        limiares["MAX_AREA_FACE"] = int(np.percentile(self.amostras_area, 95))
        limiares["MIN_AR"] = float(np.percentile(self.amostras_ar, 5))
        limiares["MAX_AR"] = float(np.percentile(self.amostras_ar, 95))

        # Guardrail: se AR colapsar (ex.: 1.0 / 1.0), abrimos uma janela razoável.
        if (limiares["MAX_AR"] - limiares["MIN_AR"]) < 0.15:
            limiares["MIN_AR"] = 0.6
            limiares["MAX_AR"] = 1.6

        # Confiança: só se houver amostras reais
        if len(self.amostras_confianca) > 0:
            limiares["MIN_CONFIANCA"] = float(np.percentile(self.amostras_confianca, 20))
        else:
            limiares["MIN_CONFIANCA"] = 0.0

        self.limiares_definidos = True

        if self.debug:
            print("[DEBUG] Autoajuste definido!")
            print(f"        MIN_AREA_FACE={limiares['MIN_AREA_FACE']} | MAX_AREA_FACE={limiares['MAX_AREA_FACE']}")
            print(f"        MIN_AR={limiares['MIN_AR']:.3f} | MAX_AR={limiares['MAX_AR']:.3f}")
            print(f"        MIN_CONFIANCA={limiares['MIN_CONFIANCA']:.3f} (amostras_conf={len(self.amostras_confianca)})")

        return limiares


def passa_filtros_geometricos(area: int, ar: float, limiares: dict) -> bool:
    """Aplica limites de área e AR (Patch 1)."""
    if area < limiares["MIN_AREA_FACE"] or area > limiares["MAX_AREA_FACE"]:
        return False
    if not (limiares["MIN_AR"] <= ar <= limiares["MAX_AR"]):
        return False
    return True


# ============================================================
# PATCH 2 — FILTRO POR CONFIANÇA (SE EXISTIR)
# ============================================================
def passa_filtro_confianca(tem_confianca: bool, confianca: float | None, limiares: dict) -> bool:
    """
    Só filtra se confidence existir.
    Se o backend não fornece confidence, não bloqueia.
    """
    if not tem_confianca:
        return True
    if confianca is None:
        return True
    return confianca >= limiares["MIN_CONFIANCA"]


# ============================================================
# PATCH 3 — PERSISTÊNCIA TEMPORAL
# ============================================================
def passa_persistencia(historico_ids, x: int, y: int, tamanho_grid: int, k_persistencia: int) -> bool:
    """
    Cria id aproximado por posição (quantização em grade) e exige persistência.
    """
    id_face = (round(x / tamanho_grid), round(y / tamanho_grid))
    historico_ids.append(id_face)

    if k_persistencia <= 1:
        return True

    return historico_ids.count(id_face) >= k_persistencia


# ============================================================
# EMOÇÃO — RECORTE DO FRAME ORIGINAL (BGR) + analyze
# ============================================================
def recortar_rosto(frame_bgr, x: int, y: int, w: int, h: int, largura: int, altura: int, pad_ratio: float = 0.15):
    """
    CORREÇÃO IMPORTANTE:
    Para evitar viés/travamento (ex.: só “angry”), recortamos do frame original.
    (Evita depender do face_crop retornado pelo DeepFace, que pode vir normalizado.)
    """
    pad = int(pad_ratio * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(largura, x + w + pad)
    y2 = min(altura, y + h + pad)

    face_crop = frame_bgr[y1:y2, x1:x2]
    return face_crop


def normalizar_resultado_analise(resultado):
    """DeepFace.analyze pode retornar dict ou list[dict]. Normaliza para dict."""
    if resultado is None:
        return None
    if isinstance(resultado, list):
        return resultado[0] if len(resultado) > 0 else None
    if isinstance(resultado, dict):
        return resultado
    return None


def analisar_emocao(face_crop_bgr):
    """
    Roda DeepFace.analyze(actions=['emotion']).
    Tentamos detector_backend='skip' (quando suportado) para evitar redetecção.
    """
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        return None

    # se muito pequeno, redimensiona para estabilizar o modelo
    if face_crop_bgr.shape[0] < 48 or face_crop_bgr.shape[1] < 48:
        face_crop_bgr = cv2.resize(face_crop_bgr, (96, 96), interpolation=cv2.INTER_LINEAR)

    try:
        resultado = DeepFace.analyze(
            img_path=face_crop_bgr,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="skip"
        )
    except Exception:
        resultado = DeepFace.analyze(
            img_path=face_crop_bgr,
            actions=["emotion"],
            enforce_detection=False
        )

    return normalizar_resultado_analise(resultado)


# ============================================================
# DESENHO / SAÍDAS
# ============================================================
def desenhar_anotacoes(frame_bgr, faces_validas: list, largura: int, altura: int):
    """Desenha bounding box e texto no frame."""
    for f in faces_validas:
        x, y, w, h = f["x"], f["y"], f["w"], f["h"]
        emocao = f["emocao"]
        score = f["score"]

        # clamp para evitar sair do frame
        x = max(0, min(x, largura - 1))
        y = max(0, min(y, altura - 1))
        w = max(0, min(w, largura - x))
        h = max(0, min(h, altura - y))

        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        texto = f"{emocao} ({score:.0f}%)"
        cv2.putText(
            frame_bgr, texto, (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )


def escrever_resumo(caminho_resumo: str, dados: dict):
    """Escreve summary.txt do Passo A."""
    with open(caminho_resumo, "w", encoding="utf-8") as f:
        f.write("=== PASSO A — Reconhecimento Facial + Emoções ===\n")
        f.write(f"Vídeo: {dados['video']}\n")
        f.write(f"Frames totais: {dados['frames_totais']}\n")
        f.write(f"Frames analisados (amostragem, step={dados['frame_step']}): {dados['frames_analisados']}\n")
        f.write(f"Total de faces detectadas (após filtros): {dados['total_faces']}\n")
        f.write("Limiares finais:\n")
        f.write(f"  MIN_AREA_FACE={dados['limiares']['MIN_AREA_FACE']}\n")
        f.write(f"  MAX_AREA_FACE={dados['limiares']['MAX_AREA_FACE']}\n")
        f.write(f"  MIN_AR={dados['limiares']['MIN_AR']:.3f}\n")
        f.write(f"  MAX_AR={dados['limiares']['MAX_AR']:.3f}\n")
        f.write(f"  MIN_CONFIANCA={dados['limiares']['MIN_CONFIANCA']:.3f}\n")
        f.write(f"  K_PERSISTENCIA={dados['k_persistencia']}\n")
        f.write(f"  TAMANHO_GRID={dados['tamanho_grid']}\n\n")

        f.write("Top emoções:\n")
        for emocao, qtd in dados["contador_emocoes"].most_common(10):
            f.write(f"- {emocao}: {qtd}\n")

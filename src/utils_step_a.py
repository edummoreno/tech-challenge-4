import os
import cv2
import numpy as np
from deepface import DeepFace
import face_recognition
import re
from threading import Thread
import queue
import time


# ============================================================
# COMPATIBILIDADE (STEP B)
# ============================================================
def abrir_video(caminho: str) -> cv2.VideoCapture:
    """
    Abre um vídeo via OpenCV (usado no Step B).
    Mantido por compatibilidade entre steps.
    """
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Vídeo não encontrado: {caminho}")

    cap = cv2.VideoCapture(caminho)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {caminho}")

    return cap


# ============================================================
# CLASSE DE LEITURA OTIMIZADA (THREADING) — STEP A
# ============================================================
class FileVideoStream:
    """
    Lê frames do vídeo em thread separada para reduzir gargalo de I/O.
    Uso:
        cap = FileVideoStream("video.mp4").start()
        while cap.more():
            ret, frame = cap.read()
    """
    def __init__(self, path: str, queue_size: int = 128):
        self.stream = cv2.VideoCapture(path)
        if not self.stream.isOpened():
            raise FileNotFoundError(f"Não foi possível abrir o vídeo: {path}")

        self.stopped = False
        self.Q = queue.Queue(maxsize=queue_size)

        self.total_frames = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.fps = float(self.stream.get(cv2.CAP_PROP_FPS)) or 30.0
        self.width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        self.height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            if not self.Q.full():
                grabbed, frame = self.stream.read()
                if not grabbed:
                    self.stopped = True
                    return
                self.Q.put(frame)
            else:
                time.sleep(0.01)

    def read(self):
        if self.more():
            return True, self.Q.get()
        return False, None

    def more(self):
        return self.Q.qsize() > 0 or not self.stopped

    def release(self):
        self.stopped = True
        self.stream.release()

    def get(self, prop_id):
        # wrapper para compatibilidade com ler_metadados_video
        if prop_id == cv2.CAP_PROP_FPS:
            return self.fps
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return self.total_frames
        return 0


# ============================================================
# UTILITÁRIOS DE ARQUIVO / VÍDEO
# ============================================================
def garantir_diretorio(caminho: str) -> None:
    os.makedirs(caminho, exist_ok=True)

def ler_metadados_video(cap) -> dict:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
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
    return DeepFace.extract_faces(
        img_path=frame_bgr,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=False,  # mais rápido
    )

def extrair_bbox_e_confianca(face_dict: dict) -> dict:
    confianca = face_dict.get("confidence", None)
    facial_area = face_dict.get("facial_area", {}) or {}
    return {
        "x": int(facial_area.get("x", 0)),
        "y": int(facial_area.get("y", 0)),
        "w": int(facial_area.get("w", 0)),
        "h": int(facial_area.get("h", 0)),
        "tem_confianca": (confianca is not None),
        "confianca": float(confianca) if confianca is not None else None,
    }

def calcular_area_e_ar(w: int, h: int) -> tuple:
    if w <= 0 or h <= 0:
        return 0, 0.0
    return w * h, (w / float(h))


# ============================================================
# AUTOAJUSTE / FILTROS (STEP A)
# ============================================================
class AutoajusteLimiar:
    def __init__(self, frames_warmup_analisados: int, area_frame: int, debug: bool = False):
        self.frames_warmup_analisados = frames_warmup_analisados
        self.area_frame = area_frame
        self.debug = debug

        self.amostras_area = []
        self.amostras_ar = []
        self.amostras_confianca = []
        self.limiares_definidos = False

    def adicionar_amostra(self, area, ar, tem_confianca, confianca):
        # evita pegar outliers muito grandes no warm-up
        if area > 0 and area < 0.6 * self.area_frame:
            self.amostras_area.append(area)
            self.amostras_ar.append(ar)
            if tem_confianca and confianca is not None:
                self.amostras_confianca.append(float(confianca))

    def pronto_para_definir(self) -> bool:
        return (not self.limiares_definidos) and (len(self.amostras_area) >= self.frames_warmup_analisados)

    def definir_limiares(self, limiares: dict) -> dict:
        if not self.amostras_area:
            return limiares

        limiares["MIN_AREA_FACE"] = int(np.percentile(self.amostras_area, 10))
        limiares["MAX_AREA_FACE"] = int(np.percentile(self.amostras_area, 95))
        limiares["MIN_AR"] = float(np.percentile(self.amostras_ar, 5))
        limiares["MAX_AR"] = float(np.percentile(self.amostras_ar, 95))

        # fallback se o vídeo tiver AR "apertado demais" e percentis colarem
        if (limiares["MAX_AR"] - limiares["MIN_AR"]) < 0.15:
            limiares["MIN_AR"] = 0.6
            limiares["MAX_AR"] = 1.6

        if len(self.amostras_confianca) > 0:
            limiares["MIN_CONFIANCA"] = float(np.percentile(self.amostras_confianca, 20))
        else:
            limiares["MIN_CONFIANCA"] = 0.0

        self.limiares_definidos = True
        if self.debug:
            print(
                f"[DEBUG] Autoajuste definido! "
                f"MIN_AREA={limiares['MIN_AREA_FACE']} | MAX_AREA={limiares['MAX_AREA_FACE']} | "
                f"AR=[{limiares['MIN_AR']:.2f},{limiares['MAX_AR']:.2f}] | "
                f"CONF_MIN={limiares['MIN_CONFIANCA']:.2f}"
            )
        return limiares


def passa_filtros_geometricos(area, ar, limiares):
    if area < limiares["MIN_AREA_FACE"] or area > limiares["MAX_AREA_FACE"]:
        return False
    if not (limiares["MIN_AR"] <= ar <= limiares["MAX_AR"]):
        return False
    return True

def passa_filtro_confianca(tem_confianca, confianca, limiares):
    if not tem_confianca or confianca is None:
        return True
    return confianca >= limiares["MIN_CONFIANCA"]

def passa_persistencia(historico_ids, x, y, grid, k):
    """
    Persistência temporal simplificada: bbox próxima deve aparecer >=k vezes no histórico.
    """
    id_face = (round(x / grid), round(y / grid))
    historico_ids.append(id_face)
    if k <= 1:
        return True
    return historico_ids.count(id_face) >= k


# ============================================================
# EMOÇÃO / IDENTIDADE
# ============================================================
def recortar_rosto(frame_bgr, x, y, w, h, largura, altura, pad_ratio=0.15):
    pad = int(pad_ratio * max(w, h))
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(largura, x + w + pad), min(altura, y + h + pad)
    return frame_bgr[y1:y2, x1:x2]

def _normalizar_resultado_analise(resultado):
    if isinstance(resultado, list):
        return resultado[0] if len(resultado) > 0 else None
    return resultado if isinstance(resultado, dict) else None

def analisar_emocao(face_crop_bgr):
    """
    Emoção robusta:
    1) tenta detector_backend="skip" (sem redetecção)
    2) fallback para analyze padrão
    """
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        return None

    # crops muito pequenos quebram/ficam instáveis em alguns modelos
    if face_crop_bgr.shape[0] < 48 or face_crop_bgr.shape[1] < 48:
        face_crop_bgr = cv2.resize(face_crop_bgr, (96, 96), interpolation=cv2.INTER_LINEAR)

    try:
        res = DeepFace.analyze(
            img_path=face_crop_bgr,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="skip",
        )
    except Exception:
        try:
            res = DeepFace.analyze(
                img_path=face_crop_bgr,
                actions=["emotion"],
                enforce_detection=False,
            )
        except Exception:
            return None

    return _normalizar_resultado_analise(res)

def carregar_banco_faces(pasta_imagens):
    encodings, names = [], []

    if not os.path.exists(pasta_imagens):
        os.makedirs(pasta_imagens, exist_ok=True)
        return [], []

    print(f"📂 Carregando identidades de: {pasta_imagens}")
    for arq in os.listdir(pasta_imagens):
        if arq.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                raw_name = os.path.splitext(arq)[0]
                nome = re.sub(r"[0-9_]+$", "", raw_name).replace("_", " ").strip().title()

                img = face_recognition.load_image_file(os.path.join(pasta_imagens, arq))
                enc = face_recognition.face_encodings(img)
                if enc:
                    encodings.append(enc[0])
                    names.append(nome)
                    print(f"  ✅ Aprendido: {nome}")
            except Exception as e:
                print(f"  ❌ Erro {arq}: {e}")

    return encodings, names

def reconhecer_identidade(face_crop, known_encodings, known_names):
    if not known_encodings or face_crop is None or face_crop.size == 0:
        return "Desconhecido"

    rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape

    enc_atual = face_recognition.face_encodings(rgb, known_face_locations=[(0, w, h, 0)])
    if not enc_atual:
        return "Desconhecido"

    dists = face_recognition.face_distance(known_encodings, enc_atual[0])
    idx = int(np.argmin(dists))
    if float(dists[idx]) < 0.55:
        return known_names[idx]
    return "Desconhecido"


# ============================================================
# DESENHO / RESUMO
# ============================================================
def desenhar_anotacoes(frame, faces, w_img, h_img):
    for f in faces:
        x, y, w, h = f["x"], f["y"], f["w"], f["h"]
        texto = f"{f.get('nome', 'Desconhecido')} | {f.get('emocao', 'unknown')}"
        cor = (0, 255, 0) if f.get("nome", "Desconhecido") != "Desconhecido" else (0, 165, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), cor, 2)
        cv2.putText(frame, texto, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

def escrever_resumo(caminho, dados):
    """
    Escreve um resumo "genérico" para Step A e também funciona para Step B/C,
    imprimindo chaves conhecidas quando existirem.
    """
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("=== RESUMO ===\n")

        # imprime chaves comuns se existirem
        for k in ["video", "frames_totais", "frames_analisados", "frame_step", "total_faces"]:
            if k in dados:
                f.write(f"{k}: {dados[k]}\n")

        # emoções (step A)
        contador_emocoes = dados.get("contador_emocoes")
        if contador_emocoes:
            f.write("\nEmocoes (contagem):\n")
            try:
                for emo, v in contador_emocoes.most_common():
                    f.write(f"- {emo}: {v}\n")
            except Exception:
                # se vier como dict simples
                for emo, v in contador_emocoes.items():
                    f.write(f"- {emo}: {v}\n")

        # limiares (step A)
        limiares = dados.get("limiares")
        if limiares:
            f.write("\nLimiar (final):\n")
            for k, v in limiares.items():
                f.write(f"- {k}: {v}\n")

        # params extras
        if "k_persistencia" in dados:
            f.write(f"\nK Persistencia: {dados['k_persistencia']}\n")
        if "tamanho_grid" in dados:
            f.write(f"Tamanho Grid: {dados['tamanho_grid']}\n")

        # dump de qualquer outra chave (pra não perder info do step B)
        known = {"video","frames_totais","frames_analisados","frame_step","total_faces",
                 "contador_emocoes","limiares","k_persistencia","tamanho_grid"}
        extras = {k:v for k,v in dados.items() if k not in known}
        if extras:
            f.write("\nOutros:\n")
            for k,v in extras.items():
                f.write(f"- {k}: {v}\n")

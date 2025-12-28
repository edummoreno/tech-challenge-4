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
# CLASSE DE LEITURA OTIMIZADA (THREADING)
# ============================================================
class FileVideoStream:
    def __init__(self, path, queue_size=128):
        """
        Lê frames do vídeo numa thread separada para evitar bloqueio de I/O.
        """
        self.stream = cv2.VideoCapture(path)
        if not self.stream.isOpened():
            raise FileNotFoundError(f"Não foi possível abrir o vídeo: {path}")
            
        self.stopped = False
        self.Q = queue.Queue(maxsize=queue_size)
        self.total_frames = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def start(self):
        # Inicia a thread de leitura
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # Loop contínuo de leitura
        while True:
            if self.stopped:
                return

            if not self.Q.full():
                grabbed, frame = self.stream.read()
                
                # Se não conseguiu ler, acabou o vídeo
                if not grabbed:
                    self.stopped = True
                    return
                
                self.Q.put(frame)
            else:
                # Se a fila está cheia, espera um pouco para não fritar a CPU
                time.sleep(0.01)

    def read(self):
        # Retorna (True, frame) ou (False, None) para manter compatibilidade com OpenCV
        if self.more():
            return True, self.Q.get()
        return False, None

    def more(self):
        # Retorna True se ainda houver frames na fila ou se o vídeo não acabou
        return self.Q.qsize() > 0 or not self.stopped

    def release(self):
        self.stopped = True
        self.stream.release()

    def get(self, prop_id):
        # Wrapper simples para propriedades básicas
        if prop_id == cv2.CAP_PROP_FPS: return self.fps
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH: return self.width
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT: return self.height
        if prop_id == cv2.CAP_PROP_FRAME_COUNT: return self.total_frames
        return 0

# ============================================================
# UTILITÁRIOS DE ARQUIVO / VÍDEO
# ============================================================
def garantir_diretorio(caminho: str) -> None:
    os.makedirs(caminho, exist_ok=True)

# (A função abrir_video antiga fica aqui como fallback, mas usaremos a classe acima)
def abrir_video(caminho: str) -> cv2.VideoCapture:
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Vídeo não encontrado: {caminho}")
    cap = cv2.VideoCapture(caminho)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir o vídeo")
    return cap

def ler_metadados_video(cap) -> dict:
    # Compatível tanto com cv2.VideoCapture quanto com FileVideoStream
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
        align=True
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
    if w <= 0 or h <= 0: return 0, 0.0
    return w * h, (w / float(h))

# ============================================================
# OTIMIZAÇÃO (Autoajuste / Filtros)
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
        if area > 0 and area < 0.6 * self.area_frame:
            self.amostras_area.append(area)
            self.amostras_ar.append(ar)
            if tem_confianca and confianca is not None:
                self.amostras_confianca.append(float(confianca))

    def pronto_para_definir(self) -> bool:
        return (not self.limiares_definidos) and (len(self.amostras_area) >= self.frames_warmup_analisados)

    def definir_limiares(self, limiares: dict) -> dict:
        if not self.amostras_area: return limiares
        limiares["MIN_AREA_FACE"] = int(np.percentile(self.amostras_area, 10))
        limiares["MAX_AREA_FACE"] = int(np.percentile(self.amostras_area, 95))
        limiares["MIN_AR"] = float(np.percentile(self.amostras_ar, 5))
        limiares["MAX_AR"] = float(np.percentile(self.amostras_ar, 95))

        if (limiares["MAX_AR"] - limiares["MIN_AR"]) < 0.15:
            limiares["MIN_AR"] = 0.6
            limiares["MAX_AR"] = 1.6

        if len(self.amostras_confianca) > 0:
            limiares["MIN_CONFIANCA"] = float(np.percentile(self.amostras_confianca, 20))
        else:
            limiares["MIN_CONFIANCA"] = 0.0

        self.limiares_definidos = True
        if self.debug:
            print(f"[DEBUG] Autoajuste definido! MinArea: {limiares['MIN_AREA_FACE']}, Conf: {limiares['MIN_CONFIANCA']:.2f}")
        return limiares

def passa_filtros_geometricos(area, ar, limiares):
    if area < limiares["MIN_AREA_FACE"] or area > limiares["MAX_AREA_FACE"]: return False
    if not (limiares["MIN_AR"] <= ar <= limiares["MAX_AR"]): return False
    return True

def passa_filtro_confianca(tem_confianca, confianca, limiares):
    if not tem_confianca or confianca is None: return True
    return confianca >= limiares["MIN_CONFIANCA"]

def passa_persistencia(historico_ids, x, y, grid, k):
    id_face = (round(x / grid), round(y / grid))
    historico_ids.append(id_face)
    if k <= 1: return True
    return historico_ids.count(id_face) >= k

# ============================================================
# EMOÇÃO E RECONHECIMENTO
# ============================================================
def recortar_rosto(frame_bgr, x, y, w, h, largura, altura, pad_ratio=0.15):
    pad = int(pad_ratio * max(w, h))
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(largura, x + w + pad), min(altura, y + h + pad)
    return frame_bgr[y1:y2, x1:x2]

def normalizar_resultado_analise(resultado):
    if isinstance(resultado, list):
        return resultado[0] if len(resultado) > 0 else None
    return resultado if isinstance(resultado, dict) else None

def analisar_emocao(face_crop_bgr):
    if face_crop_bgr is None or face_crop_bgr.size == 0: return None
    if face_crop_bgr.shape[0] < 48:
        face_crop_bgr = cv2.resize(face_crop_bgr, (96, 96))
    try:
        res = DeepFace.analyze(img_path=face_crop_bgr, actions=["emotion"], enforce_detection=False, detector_backend="skip")
        return normalizar_resultado_analise(res)
    except:
        return None

def carregar_banco_faces(pasta_imagens):
    encodings, names = [], []
    if not os.path.exists(pasta_imagens):
        os.makedirs(pasta_imagens, exist_ok=True)
        return [], []
        
    print(f"📂 Carregando identidades de: {pasta_imagens}")
    for arq in os.listdir(pasta_imagens):
        if arq.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                # Remove numeros e underline (ex: "Jim_01.jpg" -> "Jim")
                raw_name = os.path.splitext(arq)[0]
                nome = re.sub(r'[0-9_]+$', '', raw_name).replace("_", " ").strip().title()
                
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
    if not known_encodings or face_crop.size == 0: return "Desconhecido"
    rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    enc_atual = face_recognition.face_encodings(rgb, known_face_locations=[(0, w, h, 0)])
    
    if not enc_atual: return "Desconhecido"
    
    dists = face_recognition.face_distance(known_encodings, enc_atual[0])
    idx = np.argmin(dists)
    if dists[idx] < 0.55:
        return known_names[idx]
    return "Desconhecido"

# ============================================================
# DESENHO
# ============================================================
def desenhar_anotacoes(frame, faces, w_img, h_img):
    for f in faces:
        x, y, w, h = f["x"], f["y"], f["w"], f["h"]
        texto = f"{f['nome']} | {f['emocao']}"
        
        cor = (0, 255, 0) if f['nome'] != "Desconhecido" else (0, 165, 255)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), cor, 2)
        cv2.putText(frame, texto, (x, max(20, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

def escrever_resumo(caminho, dados):
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("=== RESUMO STEP A ===\n")
        f.write(f"Faces Totais: {dados['total_faces']}\n")
        for k, v in dados['contador_emocoes'].most_common():
            f.write(f"- {k}: {v}\n")
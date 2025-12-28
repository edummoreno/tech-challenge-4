import cv2
import mediapipe as mp
import numpy as np

# Inicializa MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def iniciar_pose():
    """Retorna a instância do modelo Pose configurada."""
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

def calcular_distancia(p1, p2):
    """Calcula distância euclidiana simples entre dois pontos (x,y)."""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def classificar_atividade(landmarks):
    """
    Classifica a atividade baseada na posição relativa dos pontos (landmarks).
    """
    if not landmarks:
        return "Desconhecido"

    # Pontos chave
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    l_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    r_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]

    # --- HEURÍSTICA 1: BRAÇOS LEVANTADOS (Comemoração/Susto) ---
    if l_wrist.y < nose.y and r_wrist.y < nose.y:
        if abs(l_wrist.x - nose.x) > 0.10:
            return "Bracos Levantados"

    # --- HEURÍSTICA 2: MÃO NO ROSTO (Pensativo/Espanto) ---
    dist_l_nose = calcular_distancia(l_wrist, nose)
    dist_r_nose = calcular_distancia(r_wrist, nose)
    
    if dist_l_nose < 0.15 or dist_r_nose < 0.15:
        return "Mao no Rosto/Pensativo"

    # --- HEURÍSTICA 3: BRAÇOS CRUZADOS (Defensivo) ---
    altura_media_ombro = (l_shoulder.y + r_shoulder.y) / 2
    altura_media_cotovelo = (l_elbow.y + r_elbow.y) / 2
    esta_na_altura_peito = (l_wrist.y > altura_media_ombro) and (l_wrist.y < altura_media_cotovelo)
    dist_punhos = abs(l_wrist.x - r_wrist.x)
    
    if esta_na_altura_peito and dist_punhos < 0.15:
        return "Bracos Cruzados/Fechado"

    # --- HEURÍSTICA 4: MÃOS JUNTAS (Rezando/Interação) ---
    if dist_punhos < 0.10 and l_wrist.y > nose.y:
        return "Maos Juntas"

    # --- PADRÃO ---
    return "Em Atividade (Geral)"

def desenhar_esqueleto(frame, results):
    """Desenha os ossos e pontos no frame."""
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

def desenhar_atividade(frame, atividade):
    """Escreve o nome da atividade no canto superior."""
    cv2.rectangle(frame, (5, 50), (450, 90), (0, 0, 0), -1)
    
    cor_texto = (255, 255, 0) # Ciano
    if "Levantados" in atividade:
        cor_texto = (0, 0, 255) # Vermelho se for alerta

    cv2.putText(
        frame, 
        f"Acao: {atividade}", 
        (10, 80), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.8, 
        cor_texto, 
        2
    )
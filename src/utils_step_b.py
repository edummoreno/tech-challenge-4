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

def classificar_atividade(landmarks):
    """
    Recebe os landmarks (pontos do corpo) e retorna uma string com a atividade.
    Baseado em heurísticas simples de posição.
    """
    if not landmarks:
        return "Desconhecido"

    # Extraindo pontos chave (Y é invertido: 0 é topo, 1 é base)
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    
    # HEURÍSTICA 1: Braços Levantados
    # Se os punhos estiverem acima (y menor) que o nariz
    if left_wrist.y < nose.y and right_wrist.y < nose.y:
        return "Bracos Levantados"
    
    # HEURÍSTICA 2: Mãos juntas (rezando/segurando algo)
    # Distância pequena entre os punhos e abaixo do pescoço
    dist_punhos = abs(left_wrist.x - right_wrist.x)
    if dist_punhos < 0.10 and left_wrist.y > nose.y:
        return "Maos Juntas/Interacao"

    # HEURÍSTICA 3: Padrão (Em pé / Andando)
    # Diferenciar andando de parado requer análise temporal (frames anteriores),
    # mas para simplificar frame-a-frame, assumimos "Em Atividade" se houver detecção.
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
    cv2.putText(
        frame, 
        f"Atividade: {atividade}", 
        (10, 70),  # Posição um pouco abaixo das infos do Step A
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.8, 
        (255, 255, 0), # Cor Ciano
        2
    )
import os
from collections import Counter

import cv2
from tqdm import tqdm

# DeepFace pode demorar para importar (carrega dependências)
from deepface import DeepFace


INPUT_VIDEO = os.path.join("data", "input.mp4")
OUTPUT_VIDEO = os.path.join("outputs", "stepA_annotated.mp4")
OUTPUT_SUMMARY = os.path.join("outputs", "stepA_summary.txt")

# Processa 1 frame a cada N (para ficar viável)
FRAME_STEP = 3

# Backend mais "leve" para detecção
DETECTOR_BACKEND = "opencv"  # opções comuns: "opencv", "mtcnn", "retinaface"
ENFORCE_DETECTION = False    # evita quebrar quando não achar face


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_results(results):
    """
    DeepFace.analyze pode retornar dict (1 face) ou list[dict] (múltiplas faces).
    Aqui a gente normaliza para list[dict].
    """
    if results is None:
        return []
    if isinstance(results, list):
        return results
    if isinstance(results, dict):
        return [results]
    return []


def main():
    if not os.path.exists(INPUT_VIDEO):
        raise FileNotFoundError(f"Não achei o vídeo em: {INPUT_VIDEO}")

    ensure_dir("outputs")

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Não consegui abrir o vídeo: {INPUT_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    emotion_counts = Counter()
    analyzed_frames = 0
    total_face_detections = 0

    last_faces = []  # lista de dicts: {"x","y","w","h","emotion","score"}

    pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="Passo A (faces+emoções)")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Processa só a cada N frames
        if frame_idx % FRAME_STEP == 0:
            try:
                # DeepFace espera imagem (BGR ok). Vamos analisar emoção.
                results = DeepFace.analyze(
                    img_path=frame,
                    actions=["emotion"],
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=ENFORCE_DETECTION
                )
                results = normalize_results(results)

                faces = []
                for r in results:
                    region = r.get("region", {}) or {}
                    x = int(region.get("x", 0))
                    y = int(region.get("y", 0))
                    w = int(region.get("w", 0))
                    h = int(region.get("h", 0))

                    emotion = r.get("dominant_emotion", "unknown")
                    emotion_scores = r.get("emotion", {}) or {}
                    score = float(emotion_scores.get(emotion, 0.0)) if isinstance(emotion_scores, dict) else 0.0

                    # Filtra regiões inválidas
                    if w > 0 and h > 0:
                        faces.append({"x": x, "y": y, "w": w, "h": h, "emotion": emotion, "score": score})

                last_faces = faces
                analyzed_frames += 1
            except Exception:
                # Se der algum erro pontual, não quebra o vídeo inteiro
                # Mantém last_faces do frame anterior
                pass

        # Desenha as últimas faces detectadas (mesmo nos frames "pulados")
        for f in last_faces:
            x, y, w, h = f["x"], f["y"], f["w"], f["h"]
            emotion = f["emotion"]
            score = f["score"]

            # Garante que não sai fora do frame
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = max(0, min(w, width - x))
            h = max(0, min(h, height - y))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            label = f"{emotion} ({score:.0f}%)" if score > 0 else emotion
            cv2.putText(
                frame, label, (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        # Atualiza contagens com base no frame analisado mais recente (aproximação JIT)
        # Se quiser 100% fiel, só conte quando frame_idx % FRAME_STEP == 0.
        if frame_idx % FRAME_STEP == 0:
            total_face_detections += len(last_faces)
            for f in last_faces:
                emotion_counts[f["emotion"]] += 1

        out.write(frame)
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    # Gera resumo do Passo A
    top_emotions = emotion_counts.most_common(5)

    lines = []
    lines.append("=== PASSO A: Reconhecimento facial + Emoções ===")
    lines.append(f"Vídeo: {INPUT_VIDEO}")
    lines.append(f"Total de frames no vídeo (informado): {total_frames}")
    lines.append(f"Frames efetivamente analisados (a cada {FRAME_STEP}): {analyzed_frames}")
    lines.append(f"Total de detecções de face (aprox.): {total_face_detections}")
    lines.append("")
    lines.append("Top emoções (contagem aproximada):")
    if top_emotions:
        for emo, cnt in top_emotions:
            lines.append(f"- {emo}: {cnt}")
    else:
        lines.append("- (nenhuma emoção detectada)")

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\n✅ Vídeo anotado salvo em: {OUTPUT_VIDEO}")
    print(f"✅ Resumo salvo em: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()

import cv2
import numpy as np
import os

def load_video_frames(video_path, max_frames=8, resize=(64, 64)):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, resize)

        # 🔥 Normalize + reduce memory
        frame = (frame / 255.0).astype(np.float32)

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return None

    while len(frames) < max_frames:
        frames.append(frames[-1])

    return np.array(frames, dtype=np.float32)


def load_dataset(folder, limit_per_class=3000):
    X = []
    y = []

    classes = ["stable", "unstable"]

    for label, class_name in enumerate(classes):
        class_path = os.path.join(folder, class_name)

        if not os.path.exists(class_path):
            print(f"Missing folder: {class_name}")
            continue

        files = os.listdir(class_path)[:limit_per_class]

        print(f"Loading {class_name}: {len(files)} videos")

        for file in files:
            if file.endswith(".avi"):
                video_path = os.path.join(class_path, file)

                frames = load_video_frames(video_path)

                if frames is None:
                    continue

                X.append(frames)
                y.append(label)

    print(f"Loaded total {len(X)} videos")

    return np.array(X, dtype=np.float32), np.array(y)

import cv2
import numpy as np
import os

def load_video_frames(video_path, max_frames=16, resize=(112, 112)):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, resize)
        frame = frame / 255.0
        frames.append(frame)

    cap.release()

    # Pad if less frames
    while len(frames) < max_frames:
        frames.append(frames[-1])

    return np.array(frames)


def load_dataset(folder):
    X = []
    y = []

    for label, class_name in enumerate(os.listdir(folder)):
        class_path = os.path.join(folder, class_name)

        for file in os.listdir(class_path):
            if file.endswith(".mp4"):
                video_path = os.path.join(class_path, file)
                frames = load_video_frames(video_path)

                X.append(frames)
                y.append(label)

    return np.array(X), np.array(y)
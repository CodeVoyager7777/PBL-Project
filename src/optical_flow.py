import cv2
import numpy as np

def compute_optical_flow(frames):
    magnitudes = []
    directions = []

    for i in range(1, len(frames)):
        flow = cv2.calcOpticalFlowFarneback(
            frames[i-1], frames[i],
            None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        magnitudes.append(np.mean(mag))
        directions.append(np.mean(ang))

    return magnitudes, directions

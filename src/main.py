import os
from preprocessing import load_video
from optical_flow import compute_optical_flow
from feature_extraction import extract_features
from stability_index import compute_stability_index
from visualization import plot_motion
import pandas as pd

DATA_FOLDER = "data/raw_videos"
RESULTS_FOLDER = "Results"
REPORT_FOLDER = "Results/reports"
PLOT_FOLDER = "Results/plots"

all_results = []

def process_video(video_path):
    print(f"\nProcessing: {video_path}")

    frames = load_video(video_path)
    magnitudes, directions = compute_optical_flow(frames)
    features = extract_features(magnitudes, directions)
    stability_score = compute_stability_index(features)

    video_name = os.path.basename(video_path).split(".")[0]

    plot_motion(magnitudes, video_name)

    return features, stability_score


def main():
    os.makedirs(PLOT_FOLDER, exist_ok=True)

    for file in os.listdir(DATA_FOLDER):
        if file.endswith((".mp4", ".avi", ".mov")):
            video_path = os.path.join(DATA_FOLDER, file)
            features, score = process_video(video_path)
            video_name = os.path.basename(video_path).split(".")[0]
            result = {"video_name": video_name,"stability_score": score,**features}
            all_results.append(result)

            print("Features:", features)
            print("Stability Index:", score, "/ 100")

    print("\nAll videos processed successfully.")
    df = pd.DataFrame(all_results)
    df.to_excel("Results/final_results.xlsx", index=False)
    print("Results saved to Excel")


if __name__ == "__main__":
    main()

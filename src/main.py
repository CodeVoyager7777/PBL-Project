import os
from preprocessing import load_video
from optical_flow import compute_optical_flow
from feature_extraction import extract_features
from stability_index import compute_stability_index
from visualization import plot_motion

DATA_FOLDER = "D:/Probe Trajectory Project/Data/raw_videos"
RESULTS_FOLDER = "D:/Probe Trajectory Project/Results"
REPORT_FOLDER = "D:/Probe Trajectory Project/Results/motion reports"
PLOT_FOLDER = "D:/Probe Trajectory Project/Results/plots"

def save_report(video_name, features, score):
    os.makedirs(REPORT_FOLDER, exist_ok=True)

    report_path = os.path.join(REPORT_FOLDER, f"{video_name}_report.txt")

    with open(report_path, "w") as f:
        f.write(f"Video: {video_name}\n\n")
        f.write("Extracted Motion Features:\n")
        for key, value in features.items():
            f.write(f"{key}: {value}\n")

        f.write(f"\nProbe Stability Index: {score} / 100\n")

        if score > 80:
            interpretation = "High stability."
        elif score > 50:
            interpretation = "Moderate stability with noticeable motion variation."
        else:
            interpretation = "Low stability with frequent abrupt movements."

        f.write(f"\nInterpretation: {interpretation}\n")


def process_video(video_path):
    print(f"\nProcessing: {video_path}")

    frames = load_video(video_path)
    magnitudes, directions = compute_optical_flow(frames)
    features = extract_features(magnitudes, directions)
    stability_score = compute_stability_index(features)

    video_name = os.path.basename(video_path).split(".")[0]

    # Save plot
    plot_motion(magnitudes, video_name)

    # Save report
    save_report(video_name, features, stability_score)

    return features, stability_score


def main():
    os.makedirs(PLOT_FOLDER, exist_ok=True)
    os.makedirs(REPORT_FOLDER, exist_ok=True)

    for file in os.listdir(DATA_FOLDER):
        if file.endswith((".mp4", ".avi", ".mov")):
            video_path = os.path.join(DATA_FOLDER, file)
            features, score = process_video(video_path)

            print("Features:", features)
            print("Stability Index:", score, "/ 100")

    print("\nAll videos processed successfully.")


if __name__ == "__main__":
    main()

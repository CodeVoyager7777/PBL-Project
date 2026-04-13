import os
import shutil
import pandas as pd

def sort_videos(video_folder, output_folder):
    df = pd.read_excel("Results/final_results.xlsx")

    threshold = df["stability_score"].mean()
    print(f"Using threshold: {threshold:.2f}")

    df["label"] = df["stability_score"].apply(
        lambda x: "stable" if x > threshold else "unstable"
    )

    os.makedirs("data", exist_ok=True)
    df[["video_name", "label"]].to_excel("data/labels.xlsx", index=False)
    print("labels.xlsx generated successfully!")

    stable_path = os.path.join(output_folder, "stable")
    unstable_path = os.path.join(output_folder, "unstable")

    os.makedirs(stable_path, exist_ok=True)
    os.makedirs(unstable_path, exist_ok=True)

    available_files = {f.lower(): f for f in os.listdir(video_folder)}

    for _, row in df.iterrows():
        video_name = str(row["video_name"]).strip()

        base_name = os.path.splitext(video_name)[0]
        video_name = base_name + ".avi"

        key = video_name.lower()

        if key not in available_files:
            continue

        actual_name = available_files[key]
        src_path = os.path.join(video_folder, actual_name)

        label = row["label"]

        if label == "stable":
            dest_path = os.path.join(stable_path, actual_name)
        else:
            dest_path = os.path.join(unstable_path, actual_name)

        shutil.move(src_path, dest_path)

    print("Dataset sorting completed!")


if __name__ == "__main__":
    sort_videos(
        video_folder="data/raw_videos",
        output_folder="dataset"
    )

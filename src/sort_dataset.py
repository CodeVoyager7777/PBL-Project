import os
import shutil
import pandas as pd

def sort_videos_from_excel(excel_path, video_folder, output_folder):
    # Load Excel file
    df = pd.read_excel(excel_path)

    # Create output folders
    stable_path = os.path.join(output_folder, "stable")
    unstable_path = os.path.join(output_folder, "unstable")

    os.makedirs(stable_path, exist_ok=True)
    os.makedirs(unstable_path, exist_ok=True)

    for _, row in df.iterrows():
        video_name = row["video_name"]   # column name in Excel
        label = row["label"]             # "stable" or "unstable"

        src_path = os.path.join(video_folder, video_name)

        if not os.path.exists(src_path):
            print(f"File not found: {video_name}")
            continue

        if label.lower() == "stable":
            dest_path = os.path.join(stable_path, video_name)
        else:
            dest_path = os.path.join(unstable_path, video_name)

        shutil.copy(src_path, dest_path)

    print("Dataset sorting completed!")


if __name__ == "__main__":
    sort_videos_from_excel(
        excel_path="data/labels.xlsx",
        video_folder="data/raw_videos",
        output_folder="dataset"
    )
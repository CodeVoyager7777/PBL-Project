import matplotlib.pyplot as plt
import os

PLOT_FOLDER = "D:/Probe Trajectory Project/Results/plots/"

def plot_motion(magnitudes, video_name):
    os.makedirs(PLOT_FOLDER, exist_ok=True)

    plt.figure()
    plt.plot(magnitudes)
    plt.title(f"Motion Magnitude Over Time - {video_name}")
    plt.xlabel("Frame")
    plt.ylabel("Mean Motion Magnitude")

    save_path = os.path.join(PLOT_FOLDER, f"{video_name}_motion_plot.png")
    plt.savefig(save_path)
    plt.close()

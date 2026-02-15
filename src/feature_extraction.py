import numpy as np

def extract_features(magnitudes, directions):
    features = {}

    if len(magnitudes) == 0:
        return None

    magnitudes = np.array(magnitudes)
    directions = np.array(directions)

    # Motion statistics
    features["mean_motion"] = float(np.mean(magnitudes))
    features["std_motion"] = float(np.std(magnitudes))
    features["max_motion"] = float(np.max(magnitudes))

    # Direction stability
    features["direction_variance"] = float(np.var(directions))

    # Acceleration spikes
    acceleration = np.diff(magnitudes)
    features["acceleration_std"] = float(np.std(acceleration))

    return features

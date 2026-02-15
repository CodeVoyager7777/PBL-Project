def compute_stability_index(features):
    if features is None:
        return 0

    score = 100

    # Penalize instability
    score -= features["std_motion"] * 8

    # Penalize excessive movement
    score -= features["max_motion"] * 4

    # Penalize direction inconsistency
    score -= features["direction_variance"] * 5

    # Penalize sudden acceleration
    score -= features["acceleration_std"] * 6

    score = max(0, min(100, score))
    return round(score, 2)

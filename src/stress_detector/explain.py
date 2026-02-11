def top_reason_strings(sample_row: dict, feature_importance: dict, top_k: int = 3):
    """Simple explanations based on delta sign + model feature importance."""
    pairs = []
    for feat, imp in feature_importance.items():
        if feat in sample_row:
            val = sample_row[feat]
            direction = "increased" if val > 0 else ("decreased" if val < 0 else "unchanged")
            pairs.append((imp, feat, direction))

    pairs.sort(reverse=True, key=lambda x: x[0])
    reasons = []
    for imp, feat, direction in pairs[:top_k]:
        pretty = feat.replace('_delta', '').replace('_', ' ')
        reasons.append(f"{pretty} {direction} vs baseline")
    return reasons


def early_warning(predictions, threshold: int = 3):
    """Return indices where sustained stress is detected."""
    out = []
    streak = 0
    for i, p in enumerate(predictions):
        streak = streak + 1 if p == 1 else 0
        if streak >= threshold:
            out.append(i)
    return out


def top_reason_strings(sample_row: dict, feature_importance: dict, top_k: int = 3):
    pairs = []
    for feat, imp in feature_importance.items():
        if feat in sample_row:
            val = sample_row[feat]
            direction = "increased" if val > 0 else ("decreased" if val < 0 else "unchanged")
            pretty = feat.replace("_delta", "").replace("_", " ")
            pairs.append((imp, pretty, direction))

    pairs.sort(reverse=True, key=lambda x: x[0])

    reasons = []
    seen = set()
    for imp, pretty, direction in pairs:
        if pretty in seen:
            continue
        seen.add(pretty)
        reasons.append(f"{pretty} {direction} vs baseline")
        if len(reasons) >= top_k:
            break

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

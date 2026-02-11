
import argparse
import json
import os

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


try:
    # When run as module: python -m stress_detector.train
    from .config import Config
    from .data import load_swell_csv, standardize_columns, drop_missing
    from .features import numeric_feature_columns, compute_personal_baseline, add_delta_from_baseline, make_labels
    from .model import build_model, save_model
    from .utils import ensure_dir
except ImportError:
    # When run as script: python src/stress_detector/train.py  (VS Code Play button)
    from stress_detector.config import Config
    from stress_detector.data import load_swell_csv, standardize_columns, drop_missing
    from stress_detector.features import numeric_feature_columns, compute_personal_baseline, add_delta_from_baseline, make_labels
    from stress_detector.model import build_model, save_model
    from stress_detector.utils import ensure_dir


def train(config: Config, out_dir: str = "outputs"):
    if not os.path.exists(config.data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{config.data_path}'. Put the CSV in the data/ folder as described in README.md"
        )

    df = load_swell_csv(config.data_path)
    df = standardize_columns(
        df,
        config.raw_participant_col, config.participant_col,
        config.raw_condition_col, config.condition_col
    )

    # Only drop rows missing critical columns (avoid losing data)
    df = drop_missing(df, required_cols=[config.participant_col, config.condition_col])

    exclude = [config.participant_col, config.condition_col]
    base_feats = numeric_feature_columns(df, exclude=exclude)

    baseline = compute_personal_baseline(
        df, config.participant_col, config.condition_col,
        config.neutral_code, base_feats
    )
    df2 = add_delta_from_baseline(df, baseline, config.participant_col, base_feats)

    delta_feats = [c for c in df2.columns if c.endswith('_delta')]
    feature_cols = base_feats + delta_feats

    X = df2[feature_cols]
    X = X.fillna(X.median(numeric_only=True))  # safe fill

    y = make_labels(df2, config.condition_col, config.neutral_code, config.stress_codes)

    # Robust split
    test_size = 0.2 if len(y) >= 10 else 0.5
    strat = y if (y.nunique() > 1 and y.value_counts().min() >= 2) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=config.random_state, stratify=strat
    )

    model = build_model(config.n_estimators, config.random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)

    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "models"))
    ensure_dir(os.path.join(out_dir, "reports"))

    model_path = os.path.join(out_dir, "models", "rf_stress_model.joblib")
    metadata = {
        "trained_on": os.path.basename(config.data_path),
        "feature_cols": feature_cols,
        "accuracy": acc,
    }
    save_model(model, model_path, metadata)

    metrics_path = os.path.join(out_dir, "reports", "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "confusion_matrix": cm.tolist()}, f, indent=2)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["neutral", "stressed"])
    disp.plot(values_format='d')
    plt.title("Confusion Matrix")
    fig_path = os.path.join(out_dir, "reports", "confusion_matrix.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")

    print(f"✅ Training complete. Accuracy={acc:.3f}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved plot: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Train stress detector on SWELL-KW features")
    parser.add_argument("--data", default=None, help="Path to SWELL-KW CSV (optional)")
    parser.add_argument("--out", default="outputs", help="Output folder")
    args = parser.parse_args()

    cfg = Config()
    if args.data:
        cfg.data_path = args.data

    train(cfg, args.out)


if __name__ == "__main__":
    main()

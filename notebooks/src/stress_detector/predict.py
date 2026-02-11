import argparse
import os

import pandas as pd

try:
    from .config import Config
    from .data import load_swell_csv, standardize_columns, drop_missing
    from .features import numeric_feature_columns, compute_personal_baseline, add_delta_from_baseline
    from .model import load_model
    from .explain import top_reason_strings, early_warning
except ImportError:
    from stress_detector.config import Config
    from stress_detector.data import load_swell_csv, standardize_columns, drop_missing
    from stress_detector.features import numeric_feature_columns, compute_personal_baseline, add_delta_from_baseline
    from stress_detector.model import load_model
    from stress_detector.explain import top_reason_strings, early_warning


def predict(config: Config, model_path: str, data_path: str, threshold=None) -> pd.DataFrame:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Train first (python -m stress_detector.train)")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found: {data_path}")

    model, metadata = load_model(model_path)

    df = load_swell_csv(data_path)
    df = standardize_columns(df, config.raw_participant_col, config.participant_col,
                             config.raw_condition_col, config.condition_col)

    df = drop_missing(df, required_cols=[config.participant_col])

    exclude = [c for c in [config.participant_col, config.condition_col] if c in df.columns]
    base_feats = numeric_feature_columns(df, exclude=exclude)

    if config.condition_col in df.columns:
        baseline = compute_personal_baseline(df, config.participant_col, config.condition_col,
                                             config.neutral_code, base_feats)
    else:
        baseline = pd.DataFrame()

    df2 = add_delta_from_baseline(df, baseline, config.participant_col, base_feats)

    feature_cols = metadata.get('feature_cols')
    if not feature_cols:
        delta_feats = [c for c in df2.columns if c.endswith('_delta')]
        feature_cols = base_feats + delta_feats

    X = df2[feature_cols]
    X = X.fillna(X.median(numeric_only=True))

    preds = model.predict(X)

    importances = dict(zip(feature_cols, getattr(model, 'feature_importances_', [0]*len(feature_cols))))

    t = int(threshold) if threshold is not None else config.consecutive_threshold
    warn_idx = set(early_warning(preds.tolist(), threshold=t))

    rows = []
    for i, p in enumerate(preds):
        label = 'stressed' if p == 1 else 'neutral'
        reasons = top_reason_strings(df2.iloc[i].to_dict(), importances, top_k=3)
        rows.append({
            'index': i,
            'prediction': label,
            'early_warning': i in warn_idx,
            'reasons': '; '.join(reasons)
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Predict stress + early warning + explanations")
    parser.add_argument("--model", default="outputs/models/rf_stress_model.joblib")
    parser.add_argument("--data", default=None)
    parser.add_argument("--threshold", type=int, default=None)
    parser.add_argument("--save", default="outputs/reports/predictions.csv")
    args = parser.parse_args()

    cfg = Config()
    data_path = args.data or cfg.data_path

    out = predict(cfg, args.model, data_path, threshold=args.threshold)
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    out.to_csv(args.save, index=False)

    print(out.head(10).to_string(index=False))
    print(f"\nSaved: {args.save}")



if __name__ == "__main__":
    main()

import joblib
from sklearn.ensemble import RandomForestClassifier


def build_model(n_estimators: int, random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)


def save_model(model, path: str, metadata=None) -> None:
    payload = {"model": model, "metadata": metadata or {}}
    joblib.dump(payload, path)


def load_model(path: str):
    payload = joblib.load(path)
    return payload["model"], payload.get("metadata", {})

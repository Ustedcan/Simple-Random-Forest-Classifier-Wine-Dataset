import numpy as np
from sklearn.ensemble import RandomForestClassifier

def build_model(
        n_estimators: int = 200,
        max_depth: int = 4,
        random_state: int = 42,
        n_jobs: int = -4,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs
    )

def train_model(
        model: RandomForestClassifier,
        x_train: np.ndarray,
        y_train: np.ndarray,
) -> RandomForestClassifier:
    model.fit(x_train, y_train)
    return model

def make_predictions(
        model: RandomForestClassifier,
        x_test: np.ndarray
) -> np.ndarray:
    predictions = model.predict(x_test)
    print(f"Prediciones del modelo: {predictions}")
    return predictions
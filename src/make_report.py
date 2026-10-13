import numpy as np
from sklearn.metrics import classification_report

def make_report(
        y_test: np.ndarray,
        y_pred: np.ndarray,
) -> None:
    report = classification_report(y_test, y_pred)
    print("Classification Report")
    print(report)

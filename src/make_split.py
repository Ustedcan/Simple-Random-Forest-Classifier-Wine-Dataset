import numpy as np
from sklearn.model_selection import train_test_split

from typing import Tuple

def make_split(
        x: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    if stratify is None:
        stratify = y
        
    x_train, x_test, y_train, y_test = train_test_split(
        x, # *arryas
        y, # *arrays
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    print(x_train.shape)
    return x_train, x_test, y_train, y_test
import numpy as np
import pandas as pd

from typing import Tuple

def get_x_y(
        df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    x: np.ndarray = df.iloc[:, :-1].to_numpy() # to numpy -> np.ndarray
    y: np.ndarray = df.iloc[:, -1].to_numpy()
    #print(x.shape)
    #print(y.shape)
    return x, y
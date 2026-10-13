import pandas as pd
from typing import Any

def read_csv(
        filepath_or_buffer: str,
        storage_options: dict[str, Any] 
) -> pd.DataFrame :
    df = pd.read_csv(
        filepath_or_buffer=filepath_or_buffer,
        storage_options=storage_options,
    )
    #print(df.head())
    return df
    

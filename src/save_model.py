import joblib
import s3fs

from typing import Any

def save_model(
        model: Any,
        s3_path: str,
        fs: s3fs.S3FileSystem,
) -> None:
    with fs.open(s3_path, mode="wb") as s3_file:
        joblib.dump(model, s3_file)

    print("Model saved sucessfully!")


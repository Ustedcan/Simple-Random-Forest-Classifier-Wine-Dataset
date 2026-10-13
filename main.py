import os
import pandas as pd
from dotenv import load_dotenv

from src.load_csv import read_csv
from src.get_x_y import get_x_y
from src.make_split import make_split
from src.model import build_model, train_model, make_predictions
from src.make_report import make_report

load_dotenv()

AWS_ENDPOINT_URL = os.getenv("MINIO_ENDPOINT")
AWS_ACCESS_KEY_ID = os.getenv("MINIO_USER_ROOT")
AWS_SECRET_ACCESS_KEY = os.getenv("MINIO_USER_PASSWORD")

BUCKET_NAME = os.getenv("BUCKET_NAME")
FILE_NAME = os.getenv("FILE_NAME")
S3_URL = f"s3://{BUCKET_NAME}/{FILE_NAME}"

def main() -> None:

    # Leer el file from AWS
    df = read_csv(
        filepath_or_buffer=S3_URL,
        storage_options={
            "key": AWS_ACCESS_KEY_ID,
            "secret": AWS_SECRET_ACCESS_KEY,
            "client_kwargs": {
                "endpoint_url": AWS_ENDPOINT_URL
            }
        }
    )

    # Get features, and labels -> x, y
    x, y = get_x_y(df=df)

    # X_train, y_train split
    x_train, x_test, y_train, y_test = make_split(
        x,
        y, 
        test_size=0.3,
        stratify=y,
    )

    # Instanciar el modelo, entrenar y prediciones
    model = build_model()
    fitted_model = train_model(model, x_train, y_train)
    preds = make_predictions(fitted_model, x_test)

    # Report
    make_report(y_test, preds)

if __name__ == "__main__":
    main()

import os
import s3fs
from dotenv import load_dotenv

from src.load_csv import read_csv
from src.get_x_y import get_x_y
from src.make_split import make_split
from src.model import build_model, train_model, make_predictions
from src.make_report import make_report
from save_model import save_model

load_dotenv()

BUCKET_BRONZE = os.getenv("BUCKET_BRONZE")
FILE_NAME = os.getenv("FILE_NAME")
S3_URL = f"s3://{BUCKET_BRONZE}/{FILE_NAME}"


BUCKET_GOLD = os.getenv("BUCKET_GOLD")
MODEL_FOLDER = os.getenv("MODEL_FOLDER")
MODEL_NAME = os.getenv("MODEL_NAME")

fs = s3fs.S3FileSystem()

def main() -> None:

    # Leer el file from AWS
    df = read_csv(
        filepath_or_buffer=S3_URL,
        storage_options={
            "client_kwargs": {
                "endpoint_url": os.getenv("AWS_ENDPOINT_URL")
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

    # Save model 
    model_s3_path = f"{BUCKET_GOLD}/{MODEL_FOLDER}/{MODEL_NAME}.joblib"
    save_model(model=fitted_model, s3_path=model_s3_path, fs=fs)

    # Report
    make_report(y_test, preds)

if __name__ == "__main__":
    main()

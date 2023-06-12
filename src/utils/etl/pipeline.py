from typing import Tuple

import pandas as pd

from src.constants import (
    DATA_TEST_PATH,
    DATA_TRAIN_PATH,
    MAP_PROVINCE,
)
from src.utils.etl.helper import clean_text, fill_na, filter_columns


def extract() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = pd.read_csv(DATA_TRAIN_PATH)
    df_test = pd.read_csv(DATA_TEST_PATH)
    return df_train, df_test


def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = filter_columns(df)
    df = fill_na(df)

    df[["lat", "lon"]] = df[["lon", "lat"]]
    df["province"] = df["province"].map(MAP_PROVINCE)

    df = clean_text(df)

    return df


def load_data(df: pd.DataFrame, output_path: str) -> None:
    df.to_parquet(output_path, index=False, engine="pyarrow")

from typing import Tuple

import pandas as pd
from unidecode import unidecode

from src.constants import (
    COUNTRY,
    CURRENCY,
    DATA_TEST_PATH,
    DATA_TRAIN_PATH,
    DROP_COLS,
    MAP_PROVINCE,
    OPERATION_TYPE,
    PROPERTY_TYPE,
    PROVINCE,
    RENAME_COLS,
)
from src.utils.text_cleaner import (
    remove_stopwords_punctuaction,
    replace_number_words_with_ordinals,
)


def extract() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = pd.read_csv(DATA_TRAIN_PATH)
    df_test = pd.read_csv(DATA_TEST_PATH)
    return df_train, df_test


def transform(df: pd.DataFrame, n_partitions: int = 32) -> pd.DataFrame:
    df = df.drop(DROP_COLS, axis=1)
    df = df.rename(RENAME_COLS, axis=1)

    df = df[df["country"].isin(COUNTRY)]
    df = df[df["province"].isin(PROVINCE)]
    df = df[df["property_type"].isin(PROPERTY_TYPE)]
    df = df[df["currency"].isin(CURRENCY)]
    df = df[df["operation_type"].isin(OPERATION_TYPE)]

    df.end_date = pd.to_datetime(df.end_date, errors="coerce")
    df = df.sort_values(by=["created_on"], ascending=True)
    # df = df.drop_duplicates(keep="last")

    df[["lat", "lon"]] = df[["lon", "lat"]]

    df["title"] = df["title"].str.lower().str.strip().str.replace("  ", " ")
    df["description"] = df["description"].str.lower().str.strip().str.replace("  ", " ")

    df["title"] = (
        df["title"]
        .swifter.set_dask_scheduler("processes")
        .set_npartitions(n_partitions)
        .allow_dask_on_strings(enable=True)
        .apply(lambda x: unidecode(x))
        .apply(remove_stopwords_punctuaction)
        .apply(replace_number_words_with_ordinals)
    )

    df["description"] = (
        df["description"]
        .swifter.set_dask_scheduler("processes")
        .set_npartitions(n_partitions)
        .allow_dask_on_strings(enable=True)
        .apply(lambda x: unidecode(x))
        .apply(remove_stopwords_punctuaction)
        .apply(replace_number_words_with_ordinals)
    )

    df["province"] = df["province"].map(MAP_PROVINCE)

    return df


def load_data(df: pd.DataFrame, output_path: str) -> None:
    df.to_parquet(output_path, index=False, engine="pyarrow")

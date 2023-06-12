import numpy as np
import pandas as pd

import swifter  # noqa
from unidecode import unidecode

from src.constants import COUNTRY, CURRENCY, DROP_COLS, OPERATION_TYPE, PROPERTY_TYPE, PROVINCE, RENAME_COLS
from src.utils.etl.text_cleaner import remove_stopwords_punctuaction, replace_number_words_with_ordinals


def filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(DROP_COLS, axis=1)
    df = df.rename(RENAME_COLS, axis=1)
    df = df[df["country"].isin(COUNTRY)]
    df = df[df["province"].isin(PROVINCE)]
    df = df[df["property_type"].isin(PROPERTY_TYPE)]
    df = df[df["currency"].isin(CURRENCY)]
    df = df[df["operation_type"].isin(OPERATION_TYPE)]

    return df


def fill_na(df: pd.DataFrame) -> pd.DataFrame:
    df["rooms"] = df["rooms"].map(lambda x: np.nan if x < 1 else x)
    df["bathrooms"] = df["bathrooms"].map(lambda x: np.nan if x < 1 else x)
    df["bedrooms"] = df["bedrooms"].map(lambda x: np.nan if x < 1 else x)

    df["surface_total"] = df["surface_total"].map(lambda x: np.nan if x < 15 else x)
    df["surface_covered"] = df["surface_covered"].map(lambda x: np.nan if x < 15 else x)

    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")

    df["lat"] = df["lat"].fillna(np.nan)
    df["lon"] = df["lon"].fillna(np.nan)
    mask = ((~df["lat"].isna()) & (~df["lon"].isna()))

    df.loc[mask, "lat"] = df.loc[mask, "lat"].astype(float).round(4)
    df.loc[mask, "lon"] = df.loc[mask, "lon"].astype(float).round(4)

    mask = (~df["lat"].isna()) & (df["lon"].isna())
    df.loc[mask, "lat"] = np.nan

    mask = (df["lat"].isna()) & (~df["lon"].isna())
    df.loc[mask, "lon"] = np.nan

    return df


def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    df["title"] = (
        df["title"]
        .swifter
        .allow_dask_on_strings(enable=True)
        .apply(lambda x: unidecode(x).lower().replace("  ", " ").replace(",", ".").strip())
        .apply(remove_stopwords_punctuaction)
        .apply(replace_number_words_with_ordinals)
    )

    df["description"] = (
        df["description"]
        .swifter
        .allow_dask_on_strings(enable=True)
        .apply(lambda x: unidecode(x).lower().replace("  ", " ").replace(",", ".").strip())
        .apply(remove_stopwords_punctuaction)
        .apply(replace_number_words_with_ordinals)
    )

    return df

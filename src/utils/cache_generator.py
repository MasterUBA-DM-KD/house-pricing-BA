import swifter
import pandas as pd
from src.utils.impute.impute import get_suburb


def cache_suburbs():
    df = pd.read_parquet("data/interim/df_test.parquet", engine="pyarrow")

    mask = ((~df["lat"].isna()) & (~df["lon"].isna()))

    df.loc[mask, "lat"] = df.loc[mask, "lat"].astype(float).round(4)
    df.loc[mask, "lon"] = df.loc[mask, "lon"].astype(float).round(4)

    df.loc[mask, "suburb"]= df.loc[mask, ["lat","lon"]].swifter.apply(lambda x: get_suburb(x["lat"], x["lon"]), axis=1)

    return df


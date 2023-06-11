import numpy as np
import pandas as pd
import swifter  # noqa
from unidecode import unidecode
from src.constants import (
    centro_geografico_caba,
    centro_geografico_plata,
    patterns_bedrooms,
    patterns_bathrooms,
    patterns_rooms,
    patterns_surface,
    patterns_surface_covered,
    patterns_surface_total,
)
import swifter  # noqa
from geopy import distance

from geopy.geocoders import Nominatim
from src.utils.impute.impute import search_in_text, get_suburb, get_lat_lon

geolocator = Nominatim(user_agent="GoogleV3")


def impute_pipeline(df: pd.DataFrame, n_partitions: int = 16):
    df = suburbs(df, n_partitions)
    df = bedrooms(df, n_partitions)
    df = surface(df, n_partitions)
    df = rooms(df, n_partitions)
    df = bathrooms(df, n_partitions)
    df = corrections(df, n_partitions)


def suburbs(df: pd.DataFrame, n_partitions: int) -> pd.DataFrame:
    all_neigborhoods = []
    for i in df["suburb"].dropna().unique():
        for j in i.split("/"):
            all_neigborhoods.append(unidecode(j.strip().lower()))

    for i in df["published_suburb"].dropna().unique():
        for j in i.split("/"):
            all_neigborhoods.append(unidecode(j.strip().lower()))

    all_neigborhoods = list(set(all_neigborhoods))

    ## Barrios

    df_barrios = pd.read_csv("../data/external/barrios/barrios_caba.csv")
    df_barrios["barrio"] = df_barrios["barrio"].apply(lambda x: unidecode(x.lower()))
    df_barrios.head(2)

    for i in df_barrios["barrio"].unique():
        if i not in all_neigborhoods:
            all_neigborhoods.append(i)
            all_neigborhoods.append(i.replace(".", ""))

    imputer = df["title"].apply(lambda x: x if x in all_neigborhoods else np.nan)
    df["suburb"] = df["suburb"].fillna(imputer)

    imputer = df["description"].apply(lambda x: x if x in all_neigborhoods else np.nan)
    df["suburb"] = df["suburb"].fillna(imputer)

    lat_ba, lon_ba = centro_geografico_plata[0], centro_geografico_plata[1]
    lat_cf, lon_cf = centro_geografico_caba[0], centro_geografico_caba[1]

    location_cf = geolocator.reverse(f"{str(lat_cf)}, {str(lon_cf)}", addressdetails=True)
    location_ba = geolocator.reverse(f"{str(lat_ba)}, {str(lon_ba)}", addressdetails=True)

    mask = (
        (df["published_suburb"].isna())
        & (df["suburb"].isna())
        & (df["lat"].isna())
        & (df["lon"].isna())
        & (df["province"] == "Ciudad Autonoma de Buenos Aires")
    )
    imputer = {
        "lat": lat_cf,
        "lon": lon_cf,
        "suburb": location_cf.raw["address"]["suburb"],
        "published_suburb": location_cf.raw["address"]["suburb"],
    }

    df[mask] = df[mask].fillna(imputer)

    mask = (
        (df["published_suburb"].isna())
        & (df["suburb"].isna())
        & (df["lat"].isna())
        & (df["lon"].isna())
        & (df["province"] == "Buenos Aires")
    )
    imputer = {
        "lat": lat_ba,
        "lon": lon_ba,
        "suburb": location_ba.raw["address"]["city"],
        "published_suburb": location_ba.raw["address"]["city"],
    }

    df[mask] = df[mask].fillna(imputer)

    mask = (~df["lat"].isna()) & (~df["lon"].isna()) & (df["suburb"].isna())
    df.loc[mask, "suburb"] = (
        df.loc[mask, ["lat", "lon"]]
        .swifter.set_npartitions(n_partitions)
        .allow_dask_on_strings(enable=True)
        .apply(lambda x: get_suburb(x["lat"], x["lon"]), axis=1)
    )

    mask = (~df["published_suburb"].isna()) & (df["lat"].isna()) & (df["lon"].isna())
    df.loc[mask, "lat"], df.loc[mask, "lon"] = zip(
        *df[mask]
        .swifter.set_npartitions(n_partitions)
        .allow_dask_on_strings(enable=True)
        .apply(lambda x: get_lat_lon(x["published_suburb"], x["province"]), axis=1)
    )

    mask = (~df["suburb"].isna()) & (df["lat"].isna()) & (df["lon"].isna())
    df.loc[mask, "lat"], df.loc[mask, "lon"] = zip(
        *df[mask]
        .swifter.set_npartitions(n_partitions)
        .allow_dask_on_strings(enable=True)
        .apply(lambda x: get_lat_lon(x["suburb"], x["province"]), axis=1)
    )

    mask = df["suburb"].isna()
    df.loc[mask, "suburb"] = (
        df.loc[mask, ["lat", "lon"]]
        .swifter.set_npartitions(n_partitions)
        .allow_dask_on_strings(enable=True)
        .apply(lambda x: get_suburb(x["lat"], x["lon"]), axis=1)
    )

    mask = df["published_suburb"].isna()
    df.loc[mask, "published_suburb"] = (
        df.loc[mask, ["lat", "lon"]]
        .swifter.set_npartitions(64)
        .allow_dask_on_strings(enable=True)
        .apply(lambda x: get_suburb(x["lat"], x["lon"]), axis=1)
    )

    return df


def bedrooms(df: pd.DataFrame, n_partitions: int) -> pd.DataFrame:
    mask = df["bedrooms"].isna()

    imputer = (
        df.loc[mask, ["title"]]
        .swifter.set_npartitions(16)
        .allow_dask_on_strings(enable=True)
        .apply(lambda x: search_in_text(x["title"], patterns_bedrooms), axis=1)
    )
    df.loc[mask, "bedrooms"] = imputer

    imputer = (
        df.loc[mask, ["description"]]
        .swifter.set_npartitions(16)
        .allow_dask_on_strings(enable=True)
        .apply(lambda x: search_in_text(x["description"], patterns_bedrooms), axis=1)
    )

    df.loc[mask, "bedrooms"] = imputer

    df["bedrooms"] = df["bedrooms"].fillna(df["rooms"] - 1)

    return df


def surface(df: pd.DataFrame, n_partitions: int) -> pd.DataFrame:
    df = surface_covered(df, n_partitions)
    df = surface_total(df, n_partitions)

    mask = df[df["surface_total"] < df["surface_covered"]]
    df.loc[mask.index, "surface_total"] = mask.surface_covered
    df.loc[mask.index, "surface_covered"] = mask.surface_total

    mask = (~df["surface_covered"].isna()) & (~df["rooms"].isna())
    meters_per_room = (df[mask]["surface_covered"] / df[mask]["rooms"]).mean()
    df["surface_covered"] = df["bathrooms"].fillna(np.ceil(df["rooms"] * meters_per_room))

    df["surface_covered"] = df["surface_covered"].fillna(df["surface_total"])
    df["surface_total"] = df["surface_total"].fillna(df["surface_covered"])

    return df


def surface_covered(df: pd.DataFrame, n_partitions: int) -> pd.DataFrame:
    mask = df["surface_covered"].isna()
    imputer = (
        df.loc[mask, ["title"]]
        .swifter.set_npartitions(16)
        .allow_dask_on_strings(enable=True)
        .apply(lambda x: search_in_text(x["title"], patterns_surface_covered), axis=1)
    )
    df.loc[mask, "surface_covered"] = df.loc[mask, "surface_covered"].fillna(imputer)

    imputer = (
        df.loc[mask, ["description"]]
        .swifter.set_npartitions(16)
        .allow_dask_on_strings(enable=True)
        .apply(lambda x: search_in_text(x["description"], patterns_surface_covered), axis=1)
    )

    df.loc[mask, "surface_covered"] = df.loc[mask, "surface_covered"].fillna(imputer)

    return df


def surface_total(df: pd.DataFrame, n_partitions: int) -> pd.DataFrame:
    mask = df["surface_total"].isna()

    imputer = (
        df.loc[mask, ["title"]]
        .swifter.set_npartitions(16)
        .allow_dask_on_strings(enable=True)
        .apply(lambda x: search_in_text(x["title"], patterns_surface_total), axis=1)
    )

    df.loc[mask, "surface_total"] = df["surface_total"].fillna(imputer)
    imputer = (
        df.loc[mask, ["description"]]
        .swifter.set_npartitions(16)
        .allow_dask_on_strings(enable=True)
        .apply(lambda x: search_in_text(x["description"], patterns_surface_total), axis=1)
    )
    df.loc[mask, "surface_total"] = df["surface_total"].fillna(imputer)

    mask = df["surface_total"].isna()

    imputer = (
        df.loc[mask, ["title"]]
        .swifter.set_npartitions(16)
        .allow_dask_on_strings(enable=True)
        .apply(lambda x: search_in_text(x["title"], patterns_surface), axis=1)
    )

    df.loc[mask, "surface_covered"] = imputer

    imputer = (
        df.loc[mask, ["description"]]
        .swifter.set_npartitions(16)
        .allow_dask_on_strings(enable=True)
        .apply(lambda x: search_in_text(x["description"], patterns_surface), axis=1)
    )

    df.loc[mask, "surface_covered"] = imputer

    return df


def rooms(df: pd.DataFrame, n_partitions: int) -> pd.DataFrame:
    mask = df["rooms"].isna()

    try:
        df.loc[mask, "rooms"] = (
            df.loc[mask, ["title"]]
            .swifter.set_npartitions(16)
            .allow_dask_on_strings(enable=True)
            .apply(lambda x: search_in_text(x["title"], patterns_rooms), axis=1)
        )
    except Exception:
        pass

    try:
        df.loc[mask, "rooms"] = (
            df.loc[mask, ["description"]]
            .swifter.set_npartitions(16)
            .allow_dask_on_strings(enable=True)
            .apply(lambda x: search_in_text(x["description"], patterns_rooms), axis=1)
        )

    except:
        pass

    return df


def bathrooms(df: pd.DataFrame, n_partitions: int) -> pd.DataFrame:
    mask = df["bathrooms"].isna()

    df.loc[mask, "bathrooms"] = (
        df.loc[mask, ["title"]]
        .swifter.set_npartitions(16)
        .allow_dask_on_strings(enable=True)
        .apply(lambda x: search_in_text(x["title"], patterns_bathrooms), axis=1)
    )

    df.loc[mask, "bathrooms"] = (
        df.loc[mask, ["description"]]
        .swifter.set_npartitions(16)
        .allow_dask_on_strings(enable=True)
        .apply(lambda x: search_in_text(x["description"], patterns_bathrooms), axis=1)
    )

    mask = (~df["bathrooms"].isna()) & (~df["rooms"].isna())
    bath_per_room = (df[mask]["bathrooms"] / df[mask]["rooms"]).mean()

    df["bathrooms"] = df["bathrooms"].fillna(np.ceil(df["rooms"] * bath_per_room))

    return df


def corrections(df: pd.DataFrame, n_partitions: int) -> pd.DataFrame:
    df["dist_buenos_aires"] = df.apply(
        lambda x: distance.distance((x["lat"], x["lon"]), (centro_geografico_caba[0], centro_geografico_caba[1])).km,
        axis=1,
    )
    df["province"] = df[["province", "dist_buenos_aires"]].apply(
        lambda x: "Other" if x["dist_buenos_aires"] > 70 else x["province"], axis=1
    )

    return df

from typing import List
import swifter # noqa
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from unidecode import unidecode
from src.constants import (
    CG_CABA,
    CG_PLATA,
    CG_CABA_SUBURB,
    CG_PLATA_SUBURB,
    DROP_COLS_B4_TRAIN, DUMMY_COLS, PRECIOS_MEDIALUNA, PRECIOS_PROM_BARRIO, patterns_bedrooms,
    patterns_bathrooms,
    patterns_rooms,
    patterns_surface,
    patterns_surface_covered,
    patterns_surface_total,
    TIMEOUT,
    KM_CABA,
    USER_AGENT
)
import pickle
from geopy import distance
from geopy.geocoders import Nominatim

from src.utils.etl.text_cleaner import remove_stopwords_punctuation, replace_number_words_with_ordinals
from src.utils.impute.impute import search_in_text, get_suburb

geolocator = Nominatim(user_agent=USER_AGENT, timeout=TIMEOUT)


def impute_pipeline(df: pd.DataFrame):
    df = suburbs(df)
    df = rooms(df)
    df = bedrooms(df)
    df = surface(df)
    df = bathrooms(df)
    # df = corrections(df)

    return df


def impute_brute_force(df_train, df_test):
    drop_cols = DROP_COLS_B4_TRAIN + DUMMY_COLS + ["price"]
    df_transporte = pd.read_parquet("data/processed/transporte/transporte.parquet", engine="pyarrow")
    df_hospitales = pd.read_parquet("data/processed/salud/hospitales.parquet", engine="pyarrow")
    df_medialunas = pd.read_parquet("data/external/medialunas/medialunas.parquet", engine="pyarrow")

    df_transporte = df_transporte.dropna(subset=["lat", "lon"])
    df_hospitales = df_hospitales.dropna(subset=["lat", "lon"])

    df_train = df_train[df_test.columns.tolist()]
    df_train = df_train.drop_duplicates(keep='last')
    df_train = df_train[df_train["price"] > 1000]

    df_train["is_train"] = True
    df_test["is_train"] = False

    df = pd.concat([df_train, df_test], axis=0)

    df['suburb_is_published'] = df['suburb'] == df['published_suburb']

    df["prom_price_depto"] = df["suburb"]
    df["prom_price_depto"] = df["prom_price_depto"].map(PRECIOS_PROM_BARRIO)
    df["prom_price_depto"] = df["prom_price_depto"].fillna(-1)

    df["prom_price_medialuna"] = df["suburb"]
    df["prom_price_medialuna"] = df["prom_price_medialuna"].map(PRECIOS_MEDIALUNA)
    df["prom_price_medialuna"] = df["prom_price_medialuna"].fillna(-1)

    for is_col in ["amenities", "pileta", "parrilla", "quincho", "sum", "gimnasio", "spa", "garage", "lavadero",
                   "cochera", "baulera", "patio", "jardin", "premium", "retasado", "reciclado", "nuevo"]:
        df["is_" + is_col] = df[["title", "description"]].swifter.apply(lambda x: x.str.contains(is_col).any(), axis=1)

    le = LabelEncoder()
    columnsToEncode = ["ad_type", "property_type", "operation_type"]
    for feature in columnsToEncode:
        df[feature] = le.fit_transform(df[feature])

    df = pd.concat([df, pd.get_dummies(df[DUMMY_COLS])], axis=1)

    df_train = df[df["is_train"]].drop("is_train", axis=1)
    df_test = df[~df["is_train"]].drop("is_train", axis=1)

    df_train_dropped = df_train[drop_cols]
    df_train_dropped = df_train_dropped.reset_index(drop=True)
    df_test_dropped = df_test[drop_cols]
    df_test_dropped = df_test_dropped.reset_index(drop=True)

    id_train = df_train.index
    id_test = df_test.index

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    df_train = df_train.drop(drop_cols + ["price"], axis=1)
    df_test = df_test.drop(drop_cols + ["price"], axis=1)

    assert len(df_train.columns) == len(df_test.columns)

    # imputer = KNNImputer(n_neighbors=3)
    # imputer.fit(df_train)
    #
    # with open("data/processed/imputer.pkl", "wb") as f:
    #     pickle.dump(imputer, f)

    with open("data/processed/imputer.pkl", "rb") as f:
        imputer = pickle.load(f)

    imputed_values = imputer.transform(df_test)
    df_test = pd.DataFrame(imputed_values, columns=list(df_test.columns))
    df_test["id"] = id_test

    imputed_values = imputer.transform(df_train)
    df_train = pd.DataFrame(imputed_values, columns=list(df_train.columns))
    df_train["id"] = id_train

    df_train["closest_transport"], df_train["n_transports"] = df_train[["lat", "lon"]].swifter.apply(
        lambda x: get_closest_locations(x["lat"], x["lon"], df_transporte, 1), axis=1, result_type="expand"
    )

    df_train["closest_hospital"], df_train["n_hospitals"] = df_train[["lat", "lon"]].swifter.apply(
        lambda x: get_closest_locations(x["lat"], x["lon"], df_hospitales, 3), axis=1, result_type="expand"
    )

    df_train["dist_to_closest"], df_train["n_closest"], df_train["coffe_price"], df_train["coffe_price_min"], \
        df_train["coffe_price_max"], df_train["n_notables"], df_train["n_supernotables"] = df_train[
        ["lat", "lon"]].swifter.apply(lambda x: closet_coffe_shop(x["lat"], x["lon"], df_medialunas, 1), axis=1,
                                      result_type="expand")
    df_test["closest_transport"], df_test["n_transports"] = df_test[["lat", "lon"]].swifter.apply(
        lambda x: get_closest_locations(x["lat"], x["lon"], df_transporte, 1), axis=1, result_type="expand"
    )

    df_test["closest_hospital"], df_test["n_hospitals"] = df_test[["lat", "lon"]].swifter.apply(
        lambda x: get_closest_locations(x["lat"], x["lon"], df_hospitales, 3), axis=1, result_type="expand"
    )
    df_test["dist_to_closest"], df_test["n_closest"], df_test["coffe_price"], df_test["coffe_price_min"], df_test[
        "coffe_price_max"], df_test["n_notables"], df_test["n_supernotables"] = df_test[
        ["lat", "lon"]].swifter.apply(lambda x: closet_coffe_shop(x["lat"], x["lon"], df_medialunas, 1), axis=1,
                                      result_type="expand")

    for col in df_train_dropped.columns:
        df_train[col] = df_train_dropped[col]
    for col in df_test_dropped.columns:
        df_test[col] = df_test_dropped[col]

    return df_train, df_test


def get_closest_locations(lat, lon, df_secondary, threshold):
    n_closest = 0
    list_nearest = []
    for i in df_secondary.iterrows():
        dist = distance.great_circle((i[1]["lat"], i[1]["lon"]), (lat, lon)).km
        list_nearest.append(dist)
        if dist <= threshold:
            n_closest += 1

    dist_to_closest = min(list_nearest)
    return dist_to_closest, n_closest


def closet_coffe_shop(lat, lon, df_secondary, threshold):
    n_closest = 0
    list_nearest = []
    coffe_price = []
    n_notables = 0
    n_supernotables = 0
    for i in df_secondary.iterrows():
        dist = distance.great_circle((i[1]["lat"], i[1]["lon"]), (lat, lon)).km
        list_nearest.append(dist)
        if dist <= threshold:
            n_closest += 1
            coffe_price.append(i[1]["price"])
            if i[1]["notable"]:
                n_notables += 1
            if i[1]["supernotable"]:
                n_supernotables += 1

    if len(coffe_price) > 0:
        coffe_price_min = min(coffe_price)
        coffe_price_max = max(coffe_price)
    else:
        coffe_price_min = -1
        coffe_price_max = -1

    dist_to_closest = min(list_nearest)

    return dist_to_closest, n_closest, coffe_price, coffe_price_min, coffe_price_max, n_notables, n_supernotables

def impute_suburbs_cg(df: pd.DataFrame) -> pd.DataFrame:
    lat_caba, lon_caba = CG_CABA[0], CG_CABA[1]
    lat_plata, lon_plata = CG_PLATA[0], CG_PLATA[1]

    mask = (
            (df["published_suburb"].isna())
            & (df["suburb"].isna())
            & (df["lat"].isna())
            & (df["lon"].isna())
            & (df["province"] == "Ciudad Autonoma de Buenos Aires")
    )
    imputer = {
        "lat": lat_caba,
        "lon": lon_caba,
        "suburb": CG_CABA_SUBURB,
        "published_suburb": CG_CABA_SUBURB,
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
        "lat": lat_plata,
        "lon": lon_plata,
        "suburb": CG_PLATA_SUBURB,
        "published_suburb": CG_PLATA_SUBURB,
    }

    df[mask] = df[mask].fillna(imputer)

    return df


def impute_suburbs_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
    mask = (~df["lat"].isna()) & (~df["lon"].isna()) & (df["suburb"].isna())
    df.loc[mask, "suburb"] = df.loc[mask, ["lat", "lon"]].swifter.apply(lambda x: get_suburb(x["lat"], x["lon"]), axis=1)

    return df


def impute_suburbs_cache(df: pd.DataFrame) -> pd.DataFrame:
    cache_suburbs = pd.read_csv("data/processed/cache_suburbs.csv")
    mask = ((~df["lat"].isna()) & (~df["lon"].isna()))
    df = df.merge(cache_suburbs, on=["lat", "lon"], how="left", suffixes=("", "_y"))
    df["suburb"] = df["suburb_y"]
    df["published_suburb"] = df["published_suburb_y"]
    df = df.drop(["suburb_y", "published_suburb_y"], axis=1)

    return df


def suburbs(df: pd.DataFrame) -> pd.DataFrame:
    all_suburbs = get_all_suburbs(df)
    df = impute_suburbs_from_text(df, all_suburbs)
    # print(df.isna().sum())
    # df = impute_suburbs_cg(df)
    # print(df.isna().sum())
    # df = impute_suburbs_cache(df)
    # print(df.isna().sum())
    df = impute_suburbs_lat_lon(df)
    # print(df.isna().sum())

    # mask = (~df["published_suburb"].isna()) & (df["lat"].isna()) & (df["lon"].isna())
    # df.loc[mask, "lat"], df.loc[mask, "lon"] = df.loc[mask, ["published_suburb", "province"]].swifter.apply(lambda x: get_lat_lon(x["published_suburb"], x["province"]), axis=1, result_type="expand")
    # print(df.isna().sum())

    #mask = (~df["suburb"].isna()) & (df["lat"].isna()) & (df["lon"].isna())
    #df.loc[mask, "lat"], df.loc[mask, "lon"] = df.loc[mask, ["suburb", "province"]].swifter.apply(lambda x: get_lat_lon(x["suburb"], x["province"]), axis=1, result_type="expand")
    # print(df.isna().sum())

    mask = (df["suburb"].isna()) & (~df["lat"].isna()) & (~df["lon"].isna())
    df.loc[mask, "suburb"] = df.loc[mask, ["lat", "lon"]].swifter.apply(lambda x: get_suburb(x["lat"], x["lon"]), axis=1)
    # print(df.isna().sum())

    df["published_suburb"] = df["published_suburb"].fillna(df["suburb"])
    # print(df.isna().sum())

    df["suburb"] = df["suburb"].fillna(df["published_suburb"])
    # print(df.isna().sum())


    return df


def get_all_suburbs(df: pd.DataFrame) -> List[str]:
    df_barrios = pd.read_csv("data/external/barrios/barrios_caba.csv")
    df_barrios["barrio"] = df_barrios["barrio"].swifter.apply(lambda x: unidecode(x.lower())).apply(remove_stopwords_punctuation).apply(replace_number_words_with_ordinals)

    # df_barrios = df_barrios.rename(columns={"barrio": "suburb"})

    all_suburbs = []
    for i in df["suburb"].dropna().unique():
        for j in i.split("/"):
            all_suburbs.append(unidecode(j.strip().lower()))
    for i in df["published_suburb"].dropna().unique():
        for j in i.split("/"):
            all_suburbs.append(unidecode(j.strip().lower()))
    all_suburbs = list(set(all_suburbs))
    for i in df_barrios["barrio"].unique():
        if i not in all_suburbs:
            all_suburbs.append(i)
            all_suburbs.append(i.replace(".", ""))

    return all_suburbs


def bedrooms(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["bedrooms"].isna()
    imputer = df.loc[mask, ["title"]].swifter.apply(lambda x: search_in_text(x["title"], patterns_bedrooms), axis=1)
    df.loc[mask, "bedrooms"] = imputer

    mask = df["bedrooms"].isna()
    imputer = df.loc[mask, ["description"]].swifter.apply(lambda x: search_in_text(x["description"], patterns_bedrooms), axis=1)
    df.loc[mask, "bedrooms"] = imputer

    mask = (df["bedrooms"].isna()) & (~df["rooms"].isna())

    df.loc[mask, "bedrooms"] = df.loc[mask, "bedrooms"].fillna(np.maximum(1, df.loc[mask, "rooms"] - 1))

    return df


def surface(df: pd.DataFrame) -> pd.DataFrame:
    df = surface_covered(df)
    df = surface_total(df)
    df = surface_any(df)

    mask = df[df["surface_total"] < df["surface_covered"]]
    df.loc[mask.index, "surface_total"] = mask.surface_covered
    df.loc[mask.index, "surface_covered"] = mask.surface_total

    mask = (~df["surface_covered"].isna()) & (~df["rooms"].isna())
    meters_per_room = (df[mask]["surface_covered"] / df[mask]["rooms"]).median()
    df["surface_covered"] = df["surface_covered"].fillna(np.ceil(df["rooms"] * meters_per_room))

    df["surface_covered"] = df["surface_covered"].fillna(df["surface_total"])
    df["surface_total"] = df["surface_total"].fillna(df["surface_covered"])

    return df


def surface_any(df):
    mask = df["surface_total"].isna()
    df.loc[mask, "surface_covered"] = df.loc[mask, ["description"]].swifter.apply(lambda x: search_in_text(x["description"], patterns_surface), axis=1)

    mask = df["surface_covered"].isna()
    df.loc[mask, "surface_covered"] = df.loc[mask, ["title"]].swifter.apply(lambda x: search_in_text(x["title"], patterns_surface), axis=1)
    return df


def surface_covered(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["surface_covered"].isna()
    df.loc[mask, "surface_covered"] = df.loc[mask, ["description"]].swifter.apply(lambda x: search_in_text(x["description"], patterns_surface_covered), axis=1)

    mask = df["surface_covered"].isna()
    df.loc[mask, "surface_covered"] = df.loc[mask, ["title"]].swifter.apply(lambda x: search_in_text(x["title"], patterns_surface_covered), axis=1)


    return df


def surface_total(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["surface_total"].isna()
    df.loc[mask, "surface_total"] = df.loc[mask, ["description"]].swifter.apply(lambda x: search_in_text(x["description"], patterns_surface_total), axis=1)

    mask = df["surface_total"].isna()
    df.loc[mask, "surface_total"] = (df.loc[mask, ["title"]].swifter.apply(lambda x: search_in_text(x["title"], patterns_surface_total), axis=1))

    return df


def rooms(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["rooms"].isna()

    df.loc[mask, "rooms"] = df.loc[mask, ["title"]].swifter.apply(lambda x: search_in_text(x["title"], patterns_rooms), axis=1)
    df.loc[mask, "rooms"] = df.loc[mask, ["description"]].swifter.apply(lambda x: search_in_text(x["description"], patterns_rooms), axis=1)

    mask = (df["rooms"].isna()) & (~df["title"].isna())
    df.loc[mask, "rooms"] = df.loc[mask, "title"].swifter.apply(lambda x: 1 if "monoambiente" in str(x) else np.nan)

    mask = (df["rooms"].isna()) & (~df["description"].isna())
    df.loc[mask, "rooms"] = df.loc[mask, "description"].swifter.apply(lambda x: 1 if " ambiente " in str(x) else np.nan)

    mask = (~df["rooms"].isna()) & (~df["surface_total"].isna())
    rooms_per_area = (df[mask]["rooms"] / df[mask]["surface_total"]).median()
    df["rooms"] = df["rooms"].fillna(np.ceil(df["surface_total"] * rooms_per_area))

    mask = (~df["rooms"].isna()) & (~df["surface_covered"].isna())
    rooms_per_area = (df[mask]["rooms"] / df[mask]["surface_covered"]).median()
    df["rooms"] = df["rooms"].fillna(np.ceil(df["surface_covered"] * rooms_per_area))

    return df


def bathrooms(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["bathrooms"].isna()

    df.loc[mask, "bathrooms"] = df.loc[mask, ["title"]].swifter.apply(lambda x: search_in_text(x["title"], patterns_bathrooms), axis=1)

    df.loc[mask, "bathrooms"] = df.loc[mask, ["description"]].swifter.apply(lambda x: search_in_text(x["description"], patterns_bathrooms), axis=1)

    mask = (~df["bathrooms"].isna()) & (~df["rooms"].isna())
    bath_per_room = (df[mask]["bathrooms"] / df[mask]["rooms"]).median()

    df["bathrooms"] = df["bathrooms"].fillna(np.maximum(1, np.ceil(df["rooms"] * bath_per_room)))


    return df


def corrections(df: pd.DataFrame) -> pd.DataFrame:
    df["dist_buenos_aires"] = df.swifter.apply(
        lambda x: distance.distance((x["lat"], x["lon"]), (CG_CABA[0], CG_CABA[1])).km,
        axis=1,
    )
    df["province"] = df[["province", "dist_buenos_aires"]].swifter.apply(
        lambda x: "Other" if x["dist_buenos_aires"] > KM_CABA else x["province"], axis=1
    )

    return df


def impute_suburbs_from_text(df: pd.DataFrame, all_suburbs: List[str]) -> pd.DataFrame:
    imputer = df["title"].swifter.apply(lambda x: x if x in all_suburbs else np.nan)
    df["suburb"] = df["suburb"].fillna(imputer)

    imputer = df["description"].swifter.apply(lambda x: x if x in all_suburbs else np.nan)
    df["suburb"] = df["suburb"].fillna(imputer)

    return df

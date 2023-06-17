import pandas as pd
import numpy as np
from src.utils.etl.pipeline import load_data, extract, transform
from src.utils.impute.pipeline import impute_pipeline
from geopy import distance
import swifter # noqa

import sklearn as sk
from sklearn import model_selection
from sklearn import ensemble
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from src.constants import DATA_TRAIN_PATH, DATA_TEST_PATH
import mlflow


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


def limpiar_fold(X_train, y_train, X_test):
    # TODO: limpiar los datos

    return X_train, y_train, X_test


if __name__ == "__main__":
    df_transporte = pd.read_parquet("data/processed/transporte/transporte.parquet", engine="pyarrow")
    df_hospitales = pd.read_parquet("data/processed/salud/hospitales.parquet", engine="pyarrow")

    df_transporte = df_transporte.dropna(subset=["lat", "lon"])
    df_hospitales = df_hospitales.dropna(subset=["lat", "lon"])

    if False:
        df_train, df_test = extract()

        df_train = transform(df_train)
        df_test = transform(df_test)

        load_data(df_train, "data/raw/df_train.parquet")
        load_data(df_test, "data/raw/df_test.parquet")
    else:
        df_train = pd.read_parquet("data/raw/df_train.parquet")
        df_test = pd.read_parquet("data/raw/df_test.parquet")

    # df_train = df_train.head(3000)
    # df_test = df_test.head(3000)

    if False:
        df_test = impute_pipeline(df_test)
        df_train = impute_pipeline(df_train)

        load_data(df_train, "data/interim/df_train.parquet")
        load_data(df_test, "data/interim/df_test.parquet")
    else:
        df_train = pd.read_parquet("data/interim/df_train.parquet", engine="pyarrow")
        df_test = pd.read_parquet("data/interim/df_test.parquet", engine="pyarrow")

    if False:
        df_train = df_train.drop_duplicates(keep='last')
        df_train = df_train[df_train['suburb'].isin(df_test["suburb"].unique().tolist())]

        dummy_cols = "ad_type province suburb property_type published_suburb".split()
        drop_cols = [
            "lat",
            "lon",
            "title",
            "description",
            "currency",
            "operation_type",
            "country",
            "dist_buenos_aires"
        ]

        drop_cols = drop_cols + dummy_cols


        df_train["is_train"] = True
        df_test["is_train"] = False

        df = pd.concat([df_train, df_test], axis=0)

        df['suburb_is_published'] = df['suburb'] == df['published_suburb']

        df = pd.concat([df, pd.get_dummies(df[dummy_cols])], axis=1)
        df = df.reset_index(drop=True)

        df["closest_transport"], df["n_transports"] = df.swifter.apply(
            lambda x: get_closest_locations(x["lat"], x["lon"], df_transporte, 1), axis=1, result_type="expand"
        )

        df["closest_hospital"], df["n_hospitals"] = df.swifter.apply(
            lambda x: get_closest_locations(x["lat"], x["lon"], df_hospitales, 3), axis=1, result_type="expand"
        )

        df = df.drop(drop_cols, axis=1)

        df.set_index("id", inplace=True)

        df_train = df[df["is_train"]].drop("is_train", axis=1)
        df_test = df[~df["is_train"]].drop("is_train", axis=1)

        # df_train = df_train.dropna(axis=0)
        #
        df_train = df_train.reset_index(drop=False)
        df_test = df_test.reset_index(drop=False)

        load_data(df_train, "data/processed/df_train.parquet")
        load_data(df_test, "data/processed/df_test.parquet")
    else:
        df_train = pd.read_parquet("data/processed/df_train.parquet", engine="pyarrow")
        df_test = pd.read_parquet("data/processed/df_test.parquet", engine="pyarrow")

    df_train.set_index("id", inplace=True)
    df_test.set_index("id", inplace=True)

    df_train["rooms"] = df_train["rooms"].fillna(df_test["rooms"].median())
    df_train["bedrooms"] = df_train["bedrooms"].fillna(df_test["bedrooms"].median())

    df_train = df_train[df_train["rooms"] <= df_test["rooms"].max()]
    df_train = df_train[df_train["bedrooms"] <= df_test["bedrooms"].max()]

    # print(len(df_train.isna().sum()))
    # print(len(df_test.isna().sum()))
    # print(df_train.columns)

    df_train["price"] = df_train["price"].fillna(df_train["price"].median())

    df_train = df_train[df_train["price"] > 0]
    df_train["price"] = np.log(df_train["price"])
    indices_to_keep = ~df_train.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    df_train = df_train[indices_to_keep].astype(np.float64)

    # extra_cols = ["lat", "lon", "dist_buenos_aires", "surface_total"]
    # df_train.drop(extra_cols, axis=1, inplace=True)
    # df_test.drop(extra_cols, axis=1, inplace=True)

    # df_train = df_train.fillna(0)
    # df_test = df_test.fillna(0)
    df_train["rooms"] = df_train["rooms"].fillna(df_train["rooms"].mean())
    df_train["bathrooms"] = df_train["bathrooms"].fillna(df_train["bathrooms"].mean())
    df_train["bedrooms"] = df_train["bedrooms"].fillna(df_train["rooms"]-1)

    df_train["surface_covered"] = df_train["surface_covered"].fillna(df_train["surface_total"].mean())
    df_train["surface_covered"] = df_train["surface_covered"].fillna(df_train["surface_covered"])



    df_train = df_train.drop(["surface_total"], axis=1)
    df_test = df_test.drop(["surface_total"], axis=1)

    # Datos para probar
    df_train = df_train.select_dtypes(include=['float64', 'int64', 'int32', 'int16', 'int8', 'bool'])

    X = df_train[df_train.columns.drop('price')]
    y = df_train['price']

    # Creamos el modelo
    reg = sk.ensemble.RandomForestRegressor(n_estimators=500, max_depth=5, n_jobs=-1, random_state=42)

    # Partimos en entrenamiento+prueba y validación
    X_train_test, X_val, y_train_test, y_val = sk.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

    scores_train = []
    scores_test = []
    # Validación cruzada, 10 folds, shuffle antes

    # with mlflow.sklearn.autolog() as run:

    kf = sk.model_selection.KFold(n_splits=10, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(kf.split(X_train_test, y_train_test)):
        X_train, X_test, y_train, y_test = X_train_test.iloc[train_index], X_train_test.iloc[test_index], \
        y_train_test.iloc[train_index], y_train_test.iloc[test_index]

        X_train, y_train, X_test = limpiar_fold(X_train, y_train, X_test)

        # Entrenamos el modelo
        reg.fit(X_train, y_train)

        # Predecimos en train
        y_pred = reg.predict(X_train)
        y_pred = np.exp(y_pred)

        y_train = np.exp(y_train)

        # Medimos la performance de la predicción en test
        score_train = sk.metrics.mean_squared_error(y_train, y_pred)
        score_train_mae = sk.metrics.mean_absolute_error(y_train, y_pred)
        scores_train.append(score_train)

        # Predecimos en test
        y_pred = reg.predict(X_test)
        y_pred = np.exp(y_pred)

        y_test = np.exp(y_test)

        # Medimos la performance de la predicción en test
        score_test = sk.metrics.mean_squared_error(y_test, y_pred)
        score_test_mae = sk.metrics.mean_absolute_error(y_test, y_pred)
        scores_test.append(score_test)

        print(f"{fold=}, MSE TRAIN {score_train} MSE TEST {score_test}")
        print(f"{fold=}, MAE TRAIN {score_train_mae} MAE TEST {score_test_mae}")

    print(f"Train scores mean={pd.Series(scores_train).mean()}, std={pd.Series(scores_train).std()}")
    print(f"Test scores mean={pd.Series(scores_test).mean()}, std={pd.Series(scores_test).std()}")

    ## Datos a predecir
    X = df_train[df_train.columns.drop('price')]
    y = df_train['price']
    X_prueba = df_test[df_train.columns.drop('price')]  # cuidado:

    # Entrenamos el modelo con todos los datos
    reg.fit(X, y)

    # Predecimos
    df_test['price'] = reg.predict(X_prueba)
    df_test['price'] = np.exp(df_test['price'])

    # Grabamos
    df_test['price'].to_csv('data/processed/solucion.csv', index=True)

    importances = pd.DataFrame(
        zip(df_train.columns.drop('price'), reg.feature_importances_),
        columns=["column", "feature_importance"]
    ).sort_values(by="feature_importance", ascending=False)

    importances.to_csv("data/processed/importances.csv", index=False)

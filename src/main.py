import pandas as pd
from src.utils.etl.pipeline import load_data, extract, transform
from src.utils.impute.pipeline import impute_pipeline
from geopy import distance
import swifter # noqa
from sklearn.impute import KNNImputer

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold

from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from scipy.stats import boxcox
from scipy.special import inv_boxcox

from multiprocessing import Pool
import numpy as np

def parallelize_dataframe(df, func, n_cores=10):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

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
        print(len(df_train))
        print(len(df_test))

        df_train = parallelize_dataframe(df_train, transform)
        df_test = parallelize_dataframe(df_test, transform)

        load_data(df_train, "data/raw/df_train.parquet")
        load_data(df_test, "data/raw/df_test.parquet")
    else:
        df_train = pd.read_parquet("data/raw/df_train.parquet")
        df_test = pd.read_parquet("data/raw/df_test.parquet")

    print(len(df_train))
    print(len(df_test))

    if False:
        df_train = impute_pipeline(df_train)
        load_data(df_train, "data/interim/df_train.parquet")

        df_test = impute_pipeline(df_test)
        load_data(df_test, "data/interim/df_test.parquet")
    else:
        df_train = pd.read_parquet("data/interim/df_train.parquet", engine="pyarrow")
        df_test = pd.read_parquet("data/interim/df_test.parquet", engine="pyarrow")

    print(len(df_train))
    print(len(df_test))

    if False:
        df_train = df_train[df_test.columns.tolist()]
        df_train = df_train[df_train["price"] > 1000]
        df_train = df_train.reset_index(drop=True)
        df_train["is_retasado"] = df_train[["title", "description"]].swifter.apply(
            lambda x: x.str.contains("retasado").any(), axis=1)
        df_test["is_retasado"] = df_test[["title", "description"]].swifter.apply(
            lambda x: x.str.contains("retasado").any(), axis=1)

        df_train["is_reciclado"] = df_train[["title", "description"]].swifter.apply(
            lambda x: x.str.contains("reciclado").any(), axis=1)
        df_test["is_reciclado"] = df_test[["title", "description"]].swifter.apply(
            lambda x: x.str.contains("reciclado").any(), axis=1)
        drop_cols = [
            # "lat",
            # "lon",
            "title",
            "description",
            "currency",
            "country",
        ]
        dummy_cols = "province suburb published_suburb".split()
        drop_cols = drop_cols + dummy_cols
        df_train["is_train"] = True
        df_test["is_train"] = False

        df = pd.concat([df_train, df_test], axis=0)

        df['suburb_is_published'] = df['suburb'] == df['published_suburb']

        le = LabelEncoder()
        columnsToEncode = ["ad_type", "property_type", "operation_type"]
        for feature in columnsToEncode:
            df[feature] = le.fit_transform(df[feature])

        df = pd.concat([df, pd.get_dummies(df[dummy_cols])], axis=1)
        df = df.reset_index(drop=True)
        df = df.drop(drop_cols, axis=1)

        df_train = df[df["is_train"]].drop("is_train", axis=1)
        df_test = df[~df["is_train"]].drop("is_train", axis=1)

        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)

        df_train = df_train.drop_duplicates(keep='last')
        df_train = df_train.reset_index(drop=True)

        price_train = df_train["price"]
        price_test = df_test["price"]

        id_train = df_train["id"]
        id_test = df_test["id"]

        df_train = df_train.drop(["price", "id"], axis=1)
        df_test = df_test.drop(["price", "id"], axis=1)

        assert len(df_train.columns) == len(df_test.columns)

        imputer = KNNImputer(n_neighbors=3)
        imputer.fit(df_train)

        imputed_values = imputer.transform(df_test)
        df_test = pd.DataFrame(imputed_values, columns=list(df_test.columns))
        df_test["price"] = price_test
        df_test.index = id_test

        imputed_values = imputer.transform(df_train)
        df_train = pd.DataFrame(imputed_values, columns=list(df_train.columns))
        df_train["price"] = price_train
        df_train.index = id_train

        df_train.to_parquet("data/processed/df_train.parquet")
        df_test.to_parquet("data/processed/df_test.parquet")
    else:
        df_train = pd.read_parquet("data/processed/df_train.parquet")
        df_test = pd.read_parquet("data/processed/df_test.parquet")

    df_train["rooms"] = df_train["rooms"].round(0)
    df_train["bedrooms"] = df_train["bedrooms"].round(0)

    print(len(df_train))

    q1 = df_test["surface_covered"].quantile(0)
    q3 = df_test["surface_covered"].quantile(1)

    # bottom = q1 - 1.5 * (q3 - q1)
    # top = q3 + 1.5 * (q3 - q1)

    df_train = df_train[(df_train["surface_covered"] >= q1) & (df_train["surface_covered"] <= q3)]

    q1 = df_test["surface_total"].quantile(0)
    q3 = df_test["surface_total"].quantile(1)
    df_train = df_train[(df_train["surface_total"] >= q1) & (df_train["surface_total"] <= q3)]


    # if False:
    #     df_train = df_train[df_test.columns.tolist()]
    #     df_train = df_train.drop_duplicates(keep='last')
    #     df_train = df_train[df_train['suburb'].isin(df_test["suburb"].unique().tolist())]
    #
    #     dummy_cols = "province suburb published_suburb".split()
    #     # dummy_cols = "suburb province".split()
    #     columnsToEncode = ["ad_type", "property_type", "operation_type"]
    #
    #     drop_cols = [
    #         "lat",
    #         "lon",
    #         "title",
    #         "description",
    #         "currency",
    #         # "operation_type",
    #         "country",
    #         # "dist_buenos_aires"
    #     ]
    #
    #     drop_cols = drop_cols + dummy_cols
    #
    #     df_train["is_train"] = True
    #     df_test["is_train"] = False
    #
    #     df = pd.concat([df_train, df_test], axis=0)
    #
    #     df['suburb_is_published'] = df['suburb'] == df['published_suburb']
    #
    #     le = LabelEncoder()
    #     for feature in columnsToEncode:
    #         df[feature] = le.fit_transform(df[feature])
    #
    #     df = pd.concat([df, pd.get_dummies(df[dummy_cols])], axis=1)
    #     df = df.reset_index(drop=True)
    #
    df["closest_transport"], df["n_transports"] = df.swifter.apply(
        lambda x: get_closest_locations(x["lat"], x["lon"], df_transporte, 1), axis=1, result_type="expand"
    )

    df["closest_hospital"], df["n_hospitals"] = df.swifter.apply(
        lambda x: get_closest_locations(x["lat"], x["lon"], df_hospitales, 3), axis=1, result_type="expand"
    )
    #
    #     df = df.drop(drop_cols, axis=1)
    #
    #     df.set_index("id", inplace=True)
    #
    #     df_train = df[df["is_train"]].drop("is_train", axis=1)
    #     df_test = df[~df["is_train"]].drop("is_train", axis=1)
    #
    #     df_train = df_train.reset_index(drop=False)
    #     df_test = df_test.reset_index(drop=False)
    #
    #     load_data(df_train, "data/processed/df_train.parquet")
    #     load_data(df_test, "data/processed/df_test.parquet")
    # else:
    #     df_train = pd.read_parquet("data/processed/df_train.parquet", engine="pyarrow")
    #     df_test = pd.read_parquet("data/processed/df_test.parquet", engine="pyarrow")
    #
    # if False:
    #     df_train["rooms"] = df_train["rooms"].fillna(df_test["rooms"].mean())
    #     df_train["bedrooms"] = df_train["bedrooms"].fillna(df_test["bedrooms"].mean())
    #
    df_train = df_train[(df_train["rooms"] <= df_test["rooms"].max()) & (df_train["rooms"] > df_test["rooms"].min())]
    df_train = df_train[(df_train["bedrooms"] <= df_test["bedrooms"].max()) & (df_train["bedrooms"] > df_test["bedrooms"].min())]
    #
    #     df_train["bedrooms"] = np.log(df_train["bedrooms"])
    #     df_test["bedrooms"] = np.log(df_test["bedrooms"])
    #
    #     df_train["price"] = df_train["price"].fillna(df_train["price"].mean())
    #
    #     df_train = df_train[df_train["price"] > 0]
    #     # indices_to_keep = ~df_train.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    #     # df_train = df_train[indices_to_keep].astype(np.float64)
    #
    #     # extra_cols = ["lat", "lon", "dist_buenos_aires", "surface_total"]
    #     # df_train.drop(extra_cols, axis=1, inplace=True)
    #     # df_test.drop(extra_cols, axis=1, inplace=True)
    #
    #     df_train["rooms"] = df_train["rooms"].fillna(df_train["rooms"].mean())
    #     df_train["bathrooms"] = df_train["bathrooms"].fillna(df_train["bathrooms"].mean())
    #     df_train["bedrooms"] = df_train["bedrooms"].fillna(df_train["rooms"]-1)
    #
    #     df_train["surface_covered"] = df_train["surface_covered"].fillna(df_train["surface_total"].mean())
    #     df_train["surface_covered"] = df_train["surface_covered"].fillna(df_train["surface_covered"])
    #
    #     df_train = df_train.drop(["surface_total"], axis=1)
    #     df_test = df_test.drop(["surface_total"], axis=1)
    if True:

        # Datos para probar
        df_train = df_train.select_dtypes(include=['float64', 'int64', 'int32', 'int16', 'int8', 'bool'])

        print(len(df_train))

        X = df_train[df_train.columns.drop('price')]
        y = df_train['price']

        # Creamos el modelo
        reg = RandomForestRegressor(n_estimators=500, max_depth=5, n_jobs=-1, random_state=42)

        # Partimos en entrenamiento+prueba y validación
        X_train_test, X_val, y_train_test, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        scores_train = []
        scores_test = []
        # Validación cruzada, 10 folds, shuffle antes

        # with mlflow.sklearn.autolog() as run:

        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        for fold, (train_index, test_index) in enumerate(kf.split(X_train_test, y_train_test)):
            X_train, X_test, y_train, y_test = X_train_test.iloc[train_index], X_train_test.iloc[test_index], \
            y_train_test.iloc[train_index], y_train_test.iloc[test_index]

            X_train, y_train, X_test = limpiar_fold(X_train, y_train, X_test)

            # Entrenamos el modelo
            reg.fit(X_train, y_train)

            # Predecimos en train
            y_pred = reg.predict(X_train)

            # Medimos la performance de la predicción en test
            score_train = mean_squared_error(y_train, y_pred)
            score_train_mae = mean_absolute_error(y_train, y_pred)
            scores_train.append(score_train)

            # Predecimos en test
            y_pred = reg.predict(X_test)

            # Medimos la performance de la predicción en test
            score_test = mean_squared_error(y_test, y_pred)
            score_test_mae = mean_absolute_error(y_test, y_pred)
            scores_test.append(score_test)

            print(f"{fold=}")
            print(f"MAE TRAIN: {score_train_mae} MSE TRAIN: {score_train}")
            print(f"MAE TEST: {score_test_mae} MSE TEST: {score_test}")

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

        # Grabamos
        df_test['price'].to_csv('data/processed/solucion.csv', index=True)

        importances = pd.DataFrame(
            zip(df_train.columns.drop('price'), reg.feature_importances_),
            columns=["column", "feature_importance"]
        ).sort_values(by="feature_importance", ascending=False)

        importances.to_csv("data/processed/importances.csv", index=False)

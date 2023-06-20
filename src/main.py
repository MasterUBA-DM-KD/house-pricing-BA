import pandas as pd
from src.utils.etl.pipeline import load_data, extract, transform
from src.utils.impute.pipeline import impute_pipeline, impute_brute_force
import swifter # noqa
import pickle
from multiprocessing import Pool
from sklearn.decomposition import PCA
import numpy as np
from src.constants import PRICE_M2, DROP_COLS_B4_TRAIN, DUMMY_COLS
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error


def parallelize_dataframe(df, func, n_cores=10):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def limpiar_fold(X_train, y_train, X_test):
    # TODO: limpiar los datos

    return X_train, y_train, X_test


if __name__ == "__main__":
    if False:
        df_train, df_test = extract()
        df_train = parallelize_dataframe(df_train, transform)
        df_test = parallelize_dataframe(df_test, transform)

        load_data(df_train, "data/raw/df_train.parquet")
        load_data(df_test, "data/raw/df_test.parquet")
    else:
        df_train = pd.read_parquet("data/raw/df_train.parquet")
        df_test = pd.read_parquet("data/raw/df_test.parquet")
        df_train = df_train.set_index("id")
        df_test = df_test.set_index("id")

    if False:
        df_train = impute_pipeline(df_train)
        df_test = impute_pipeline(df_test)

        load_data(df_train, "data/interim/df_train.parquet")
        load_data(df_test, "data/interim/df_test.parquet")
    else:
        df_train = pd.read_parquet("data/interim/df_train.parquet", engine="pyarrow")
        df_test = pd.read_parquet("data/interim/df_test.parquet", engine="pyarrow")
        df_train = df_train.set_index("id")
        df_test = df_test.set_index("id")

    if False:
        df_train, df_test = impute_brute_force(df_train, df_test)

        df_train.to_parquet("data/processed/df_train.parquet", engine="pyarrow")
        df_test.to_parquet("data/processed/df_test.parquet", engine="pyarrow")
    else:
        df_train = pd.read_parquet("data/processed/df_train.parquet", engine="pyarrow")
        df_test = pd.read_parquet("data/processed/df_test.parquet", engine="pyarrow")
        df_train = df_train.set_index("id")
        df_test = df_test.set_index("id")

    if True:
        # df_train["rooms"] = df_train["rooms"].round(0)
        # df_train["bedrooms"] = df_train["bedrooms"].round(0)

        print(len(df_train))

        df_train = df_train[df_train["rooms"] <= df_test["rooms"].max()]
        df_train = df_train[df_train["bedrooms"] <= df_test["bedrooms"].max()]

        df_train["rooms"] = df_train["rooms"].round(0)
        df_train["bedrooms"] = df_train["bedrooms"].round(0)

        for i in ["guardacoche", "semipiso", "semi piso", "impecable", "privado", "expensas", "luminoso", "luz", "sol", "parrilla", "pileta", "piscina", "cochera", "cocheras", "cochera fija", "cochera cubierta", "cochera descubierta", "cochera semicubierta", "cochera doble", "cochera triple", "cochera individual", "cochera privada", "cochera subterranea", "cochera subterránea", "cochera subterraneo", "cochera subterráneo", "cochera techada", "cochera techada", "cochera cubierta", "cochera cubierto", "coche"]:
            df_train["is_"+i] = df_train["title"].str.contains(i).astype(int)
            df_train["is_"+i] = df_train["description"].str.contains(i).astype(int)
            df_test["is_"+i] = df_test["title"].str.contains(i).astype(int)
            df_test["is_"+i] = df_test["description"].str.contains(i).astype(int)

        df_train = df_train.reset_index(drop=False)

        # mask = df_train["id"].isin([652530, 651150, 651152, 652530])
        # df_train.loc[mask, "lat"], df_train.loc[mask, "lon"] = -34.61448064637565, -58.44638197269814

        # mask = df_train["id"].isin([989132])
        # df_train.loc[mask, "lat"], df_train.loc[mask, "lon"] = -34.611899576485676, -58.362587801243954
        #
        # mask = df_train["id"].isin([296311, 817492])
        # df_train.loc[mask, "lat"], df_train.loc[mask, "lon"] = -34.5886245902084, -58.39162131321634
        #
        # mask = df_train["id"].isin([626300])
        # df_train.loc[mask, "lat"], df_train.loc[mask, "lon"] = -34.5786532716302, -58.426567290655626

        df_train = df_train.set_index("id")

        # q1 = df_test["surface_covered"].quantile(0)
        # q3 = df_test["surface_covered"].quantile(1)
        # df_train = df_train[(df_train["surface_covered"] >= q1) & (df_train["surface_covered"] <= 1.5*q3)]
        #
        # q1 = df_test["surface_total"].quantile(0)
        # q3 = df_test["surface_total"].quantile(1)
        # df_train = df_train[(df_train["surface_total"] >= q1) & (df_train["surface_total"] <= 1.5*q3)]

        df_train["price"] = np.sqrt(np.sqrt(df_train["price"]))
        #
        # df_train = df_train.reset_index(drop=False)
        # df_test = df_test.reset_index(drop=False)

        # df_train["price_m2"] = np.maximum(1000000, df_train["price"] / np.maximum(df_train["surface_covered"], df_train["surface_total"]))
        df_train["price_m2"] = df_train["suburb"].map(PRICE_M2)
        df_train["price_m2"] = df_train["price_m2"].fillna(-1)

        df_test["price_m2"] = df_test["suburb"].map(PRICE_M2)
        df_test["price_m2"] = df_test["price_m2"].fillna(-1)

        df_train = df_train.drop(DROP_COLS_B4_TRAIN + DUMMY_COLS + ["lat", "lon"], axis=1)
        df_test = df_test.drop(DROP_COLS_B4_TRAIN + DUMMY_COLS + ["lat", "lon"], axis=1)

        df_train.to_parquet("data/processed/df_train_b4_train.parquet", engine="pyarrow")
        df_test.to_parquet("data/processed/df_test_b4_train.parquet", engine="pyarrow")
    else:
        df_train = pd.read_parquet("data/processed/df_train_b4_train.parquet", engine="pyarrow")
        df_test = pd.read_parquet("data/processed/df_test_b4_train.parquet", engine="pyarrow")
        df_train = df_train.set_index("id")
        df_test = df_test.set_index("id")


    if True:
        # Datos para probar
        df_train = df_train.select_dtypes(include=['float64', 'int64', 'int32', 'int16', 'int8', 'bool'])

        print(len(df_train))

        X = df_train[df_train.columns.drop('price')]
        y = df_train['price']


        # Creamos el modelo
        # reg = RandomForestRegressor(n_estimators=500, max_depth=2, n_jobs=-1, random_state=42)



        # Partimos en entrenamiento+prueba y validación
        X_train_test, X_val, y_train_test, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        scores_train = []
        scores_test = []
        # Validación cruzada, 10 folds, shuffle antes

        # with mlflow.sklearn.autolog() as run:

        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        param_grid = {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [2, 3, 4, 5, 6, 7, 8, 9, 10],
            "min_samples_leaf": [1, 2, 3, 4, 5],
            "max_features": ["auto", "sqrt", "log2"],
            "bootstrap": [True, False],
            "criterion": ["mse", "mae"],
        }
        model = RandomForestRegressor(n_jobs=-1, random_state=42)

        grid_search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            scoring="neg_mean_absolute_error",
            cv=kf, n_jobs=-1,
            verbose=1,
            random_state=42,
            refit=True
        )

        grid_search.fit(X, y)
        reg = grid_search.best_estimator_

        with open("models/rf.pkl", "wb") as f:
            pickle.dump(reg, f)


        for fold, (train_index, test_index) in enumerate(kf.split(X_train_test, y_train_test)):
            X_train, X_test, y_train, y_test = X_train_test.iloc[train_index], X_train_test.iloc[test_index], \
            y_train_test.iloc[train_index], y_train_test.iloc[test_index]


            X_train, y_train, X_test = limpiar_fold(X_train, y_train, X_test)

            # Entrenamos el modelo
            reg.fit(X_train, y_train)

            # Predecimos en train
            y_pred = reg.predict(X_train)

            y_pred = np.power(y_pred, 4)
            y_train = np.power(y_train, 4)

            # Medimos la performance de la predicción en test
            score_train = mean_squared_error(y_train, y_pred)
            score_train_mae = mean_absolute_error(y_train, y_pred)
            scores_train.append(score_train)

            # Predecimos en test
            y_pred = reg.predict(X_test)

            y_pred = np.power(y_pred, 4)
            y_test = np.power(y_test, 4)

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
        df_test["price"] = np.power(df_test["price"], 4)

        # Grabamos
        df_test['price'].to_csv('data/processed/solucion.csv', index=True)

        importances = pd.DataFrame(
            zip(X_prueba.columns, reg.feature_importances_),
            # zip(df_train.columns.drop('price'), reg.feature_importances_),
            columns=["column", "feature_importance"]
        ).sort_values(by="feature_importance", ascending=False)

        # importances.to_csv("data/processed/importances.csv", index=False)

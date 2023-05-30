from typing import Tuple

import pandas as pd


def extract() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = pd.read_csv("data/properati_entrenamiento.csv")
    df_test = pd.read_csv("data/properati_a_predecir.parquet")
    return df_train, df_test

def transform()
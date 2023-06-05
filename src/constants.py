DATA_TRAIN_PATH = "data/properati_entrenamiento.csv"
DATA_TEST_PATH = "data/properati_a_predecir.parquet"


RENAME_COLS = {
    "l1": "country",
    "l2": "province",
    "l3": "city_neighorhood",
    "l4": "published_neigborhood",
}

PROVINCE = [
    "Capital Federal",
    "Bs.As. G.B.A. Zona Sur",
    # "Bs.As. G.B.A. Zona Oeste",
    # "Bs.As. G.B.A. Zona Norte"
]

PROPERTY_TYPE = ["PH", "Departamento"]
COUNTRY = ["Argentina"]
DROP_COLS = ["l5", "l6"]
CURRENCY = ["USD"]
OPERATION_TYPE = ["Venta"]


centro_geografico_caba = [-34.61448064637565, -58.44638197269814]
centro_geografico_plata = [-34.92137784728068, -57.95455274162061]

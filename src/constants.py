DATA_TRAIN_PATH = "data/processed/properati_entrenamiento.csv"
DATA_TEST_PATH = "data/processed/properati_a_predecir.parquet"


KM_CABA = 100
USER_AGENT= "Googlev3"

RENAME_COLS = {
    "l1": "country",
    "l2": "province",
    "l3": "suburb",
    "l4": "published_suburb",
}

PROVINCE = [
    "Capital Federal",
    "Bs.As. G.B.A. Zona Sur",
]

MAP_PROVINCE = {"Capital Federal": "Ciudad Autonoma de Buenos Aires", "Bs.As. G.B.A. Zona Sur": "Buenos Aires"}


PROPERTY_TYPE = ["PH", "Departamento"]
COUNTRY = ["Argentina"]
DROP_COLS = ["l5", "l6", "price_period", "start_date", "end_date", "created_on"]
CURRENCY = ["USD"]
OPERATION_TYPE = ["Venta"]

TIMEOUT = 30

CG_CABA = [-34.61448064637565, -58.44638197269814]
CG_PLATA = [-34.92137784728068, -57.95455274162061]

CG_CABA_SUBURB = "Unknown"
CG_PLATA_SUBURB = "Unknown"


l = []
l0 = []
l1 = ["m2", "metro", "metros", "mts", "mts2", "mt2", "mt", "mtrs", "mtr"]
l2 = ["cubierto", "cubiertos", "cubierta", "cubiertas", "tot", "total", "totales", "cub"]
for i in l1:
    for j in l2:
        l.append(f"{i} {j}")
        l0.append(f"{j} {i}")
patterns_surface_total = [
    fr"\b(\d+(?:\.\d*)?)(?:\s)?(?:{'|'.join(l)})\b",
    rf"\b(?:{'|'.join(l)})(?:\s)?(\d+(?:\.\d*)?)\b",
    rf"\b(\d+(?:\.\d*)?)(?:\s)?(?:{'|'.join(l0)})\b",
    rf"\b(?:{'|'.join(l0)})(?:\s)?(\d+(?:\.\d*)?)\b",
]

l = []
l0 = []
l2 = ["descubierto", "descubiertos", "descubierta", "descubiertas", "desc"]
for i in l1:
    for j in l2:
        l.append(f"{i} {j}")
        l0.append(f"{j} {i}")
patterns_surface_covered = [
    rf"\b(\d+(?:\.\d*)?)(?:\s)?(?:{'|'.join(l)})\b",
    rf"\b(?:{'|'.join(l)})(?:\s)?(\d+(?:\.\d*)?)\b",
    rf"\b(\d+(?:\.\d*)?)(?:\s)?(?:{'|'.join(l0)})\b",
    rf"\b(?:{'|'.join(l0)})(?:\s)?(\d+(?:\.\d*)?)\b",
]

l = ["m2", "metro", "metros", "mts", "mts2", "mt2", "mt", "mtrs", "mtr"]
patterns_surface = [
    rf"\b(\d+(?:\.\d*)?)(?:\s)?(?:{'|'.join(l)})\b",
    rf"\b(?:{'|'.join(l)})(?:\s)?(\d+(?:\.\d*)?)\b"
]

patterns_rooms = [
    r"\b([1-9])\s?(?:amb|ambientes|ambiente)\b",
    r"\b(?:amb|ambientes|ambiente)\s?([1-9])\b"
]

patterns_bathrooms = [
    r"\b([1-9])\s?(?:bano|banos|lavabo|lavabos|retrete|retretes|toilet|toilets)\b",
    r"\b(?:bano|banos|lavabo|lavabos|retrete|retretes|toilet|toilets)\s*?([1-9])\b",
]

patterns_bedrooms = [
    r"\b([1-9])\s?(?:habitacion|habitaciones|alcoba|alcobas|cuarto|cuartos|pieza|piezas|suite|suites)\b",
    r"\b(?:habitacion|habitaciones|alcoba|alcobas|cuarto|cuartos|pieza|piezas|suite|suites)\s*?([1-9])\b",
]

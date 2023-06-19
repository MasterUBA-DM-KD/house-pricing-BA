DATA_TRAIN_PATH = "data/raw/properati_entrenamiento.csv"
DATA_TEST_PATH = "data/raw/properati_a_predecir.csv"


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

NUMBER_WORDS = {
    "cero": "0",
    "uno": "1",
    "dos": "2",
    "tres": "3",
    "cuatro": "4",
    "cinco": "5",
    "seis": "6",
    "siete": "7",
    "ocho": "8",
    "nueve": "9",
    "diez": "10",
}


PRECIOS_PROM_BARRIO = {
    'almagro': 2864.25,
    'balvanera': 2425.0,
    'barracas': 2466.25,
    'belgrano': 3771.75,
    'boca': 1884.5,
    'boedo': 2513.0,
    'caballito': 3072.5,
    'chacarita': 2811.3333333333,
    'coghlan': 3373.25,
    'colegiales': 3512.25,
    'constitucion': 1803.5,
    'flores': 2571.875,
    'floresta': 2082.0,
    'liniers': 2494.0,
    'mataderos': 2709.6,
    'monserrat': 2244.25,
    'monte castro': 2402.3333333333,
    'nunez': 3617.75,
    'palermo': 3840.75,
    'parque avellaneda': 1899.6666666667,
    'parque chacabuco': 2574.0,
    'parque patricios': 2060.25,
    'puerto madero': 6121.0,
    'recoleta': 3704.75,
    'retiro': 3150.75,
    'saavedra': 3101.4285714286,
    'san cristobal': 2339.5714285714,
    'san nicolas': 2453.8,
    'san telmo': 2367.0,
    'velez sarsfield': 2194.75,
    'villa crespo': 2960.625,
    'villa del parque': 2621.3333333333,
    'villa devoto': 2859.25,
    'villa general mitre': 2339.6666666667,
    'villa lugano': 1289.0,
    'villa luro': 2507.8333333333,
    'villa ortuzar': 3298.0,
    'villa pueyrredon': 2842.6666666667,
    'villa santa rita': 2461.6666666667,
    'villa urquiza': 3335.875
}

PRECIOS_MEDIALUNA = {
    "almagro": 620.0,
    "balvanera": 235.0,
    "belgrano": 300.0,
    "boedo": 350.0,
    "caballito": 612.5,
    "crucecita": 390.0,
    "flores": 300.0,
    "monserrat": 520.0,
    "nunez": 565.0,
    "palermo": 345.0,
    "parque avellaneda": 300.0,
    "recoleta": 620.0,
    "retiro": 250.0,
    "san nicolas": 305.0,
    "san telmo": 300.0,
    "vicente lopez": 360.0,
    "villa crespo": 380.0,
    "villa urquiza": 380.0
}

DUMMY_COLS = "province suburb published_suburb".split()

DROP_COLS_B4_TRAIN = [
            # "lat",
            # "lon",
            "title",
            "description",
            "currency",
            "country",
        ]

PRICE_M2 = {
    "abasto": 2719.0920198038,
    "agronomia": 3884.1734166556,
    "almagro": 3006.4455089665,
    "almirante brown": 2725.9207059327,
    "avellaneda": 2189.0524558544,
    "balvanera": 2330.016070901,
    "barracas": 2881.4704331625,
    "barrio norte": 3375.6074693302,
    "belgrano": 4493.8131254499,
    "berazategui": 2137.5600685445,
    "boca": 1900.5924475914,
    "boedo": 2784.3197946077,
    "caballito": 3295.4851385852,
    "canitas": 4902.9909074532,
    "canuelas": 1463.183549654,
    "catalinas": 2353.8319141059,
    "centro microcentro": 2752.8286461331,
    "chacarita": 3051.7862266382,
    "coghlan": 4424.4246123116,
    "colegiales": 3423.1398525849,
    "congreso": 2542.8710673708,
    "constitucion": 2018.6534068436,
    "distrito audiovisual": 3000.0,
    "esteban echeverria": 1771.1559835936,
    "ezeiza": 2029.007424208,
    "florencio varela": 1703.2673351016,
    "flores": 3178.4623174425,
    "floresta": 2281.3965619685,
    "liniers": 2663.5680795916,
    "lomas zamora": 3302.4324310298,
    "mataderos": 2989.1282800451,
    "monserrat": 2456.2184271157,
    "monte castro": 2675.261420877,
    "nan": 2672.2681302041,
    "nunez": 4556.5516940933,
    "palermo": 4343.0623841269,
    "parque avellaneda": 2280.0731350589,
    "parque centenario": 2800.6465331095,
    "parque chacabuco": 2617.520710336,
    "parque chas": 2918.1226047017,
    "parque patricios": 2044.9221981192,
    "paternal": 2960.4917667498,
    "plata": 2221.5818414651,
    "pompeya": 2194.4445456025,
    "presidente peron": 1082.6379803716,
    "puerto madero": 10795.1053244831,
    "quilmes": 2572.4096892655,
    "recoleta": 3755.839804706,
    "retiro": 3427.0662416203,
    "saavedra": 3786.2350638272,
    "san cristobal": 2931.1396522292,
    "san nicolas": 2843.5434980654,
    "san telmo": 2936.5074565198,
    "san vicente": 1405.00208538,
    "tribunales": 2726.8993564475,
    "velez sarsfield": 2309.6600115958,
    "versalles": 2430.6080552233,
    "villa crespo": 3008.6663289693,
    "villa devoto": 3623.6781605149,
    "villa general mitre": 2539.7051440001,
    "villa lugano": 2242.1799924309,
    "villa luro": 3002.0679765473,
    "villa ortuzar": 4075.2247177118,
    "villa parque": 2942.2444766749,
    "villa pueyrredon": 2989.0795592819,
    "villa real": 2477.9405354551,
    "villa riachuelo": 1731.9174377691,
    "villa santa rita": 2403.7378929932,
    "villa soldati": 1462.244239691,
    "villa urquiza": 3418.4064794794
  }
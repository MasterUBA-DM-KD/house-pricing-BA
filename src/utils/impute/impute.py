import os
import re
import numpy as np
from geopy.geocoders import Nominatim
from typing import Tuple, Union

from src.constants import TIMEOUT, USER_AGENT

from src.utils.impute.helper import can_convert_to_int

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
geolocator = Nominatim(user_agent=USER_AGENT, timeout=TIMEOUT)


def get_suburb(lat: float, lon: float) -> str:
    location = geolocator.reverse(f"{str(lat)}, {str(lon)}", addressdetails=True, timeout=TIMEOUT)
    if location is None:
        return np.nan

    for i in location.raw["address"].keys():
        if i in ["suburb", "town", "city", "village", "municipality"]:
            return location.raw["address"][i]


def get_lat_lon(suburb: str, province: str) -> Tuple[float, float]:
    location = geolocator.geocode(f"{suburb}, {province}", addressdetails=True, timeout=TIMEOUT)
    if location is None:
        return np.nan, np.nan
    return location.latitude, location.longitude


def search_in_text(text: str, patterns: list) -> Union[int, float]:
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if len(matches) > 0:
            if can_convert_to_int(matches[0]):
                return int(matches[0])
            else:
                return float(matches[0])
        else:
            return np.nan

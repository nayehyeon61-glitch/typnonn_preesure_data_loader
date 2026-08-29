from __future__ import annotations

import numpy as np


def wrapped_lon_delta(lon: np.ndarray | float, reference: float):
    return (np.asarray(lon) - reference + 180.0) % 360.0 - 180.0


def distance_bearing_km(lat1: float, lon1: float, lat2, lon2):
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(wrapped_lon_delta(lon2, lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    distance = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    bearing = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    return distance, bearing


def relative_xy_km(lat1: float, lon1: float, lat2, lon2):
    distance, bearing = distance_bearing_km(lat1, lon1, lat2, lon2)
    angle = np.radians(bearing)
    return distance * np.sin(angle), distance * np.cos(angle)


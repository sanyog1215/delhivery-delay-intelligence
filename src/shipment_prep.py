"""Shared data preparation for the Delhivery delay intelligence capstone."""
from __future__ import annotations

import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity", "label_print_batch", "scanner_build_no", "tape_roll_no",
    "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]
MEDIAN_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]


def clean_shipments(raw: pd.DataFrame, medians: dict | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    """Clean raw historical or incoming shipments and return (clean dataframe, fill medians)."""
    clean = raw.copy()
    clean = clean.drop_duplicates().copy()
    clean["service_type"] = clean["service_type"].astype("string").str.strip().str.title()
    clean["payment_mode"] = clean["payment_mode"].fillna("Unknown")
    clean["declared_value_inr"] = pd.to_numeric(
        clean["declared_value"].astype("string").str.replace("₹", "", regex=False).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )
    clean = clean.drop(columns=["declared_value"])
    clean["dispatch_time"] = pd.to_datetime(clean["dispatch_time"], errors="coerce")
    clean.loc[clean["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    if medians is None:
        fill_values = {col: float(clean[col].median()) for col in MEDIAN_COLUMNS}
    else:
        missing = set(MEDIAN_COLUMNS) - set(medians)
        if missing:
            raise ValueError(f"Missing median values for: {sorted(missing)}")
        fill_values = {col: float(medians[col]) for col in MEDIAN_COLUMNS}
    for col in MEDIAN_COLUMNS:
        clean[col] = pd.to_numeric(clean[col], errors="coerce").astype(float).fillna(fill_values[col])
    return clean, fill_values


def build_features(clean: pd.DataFrame, hubs: pd.DataFrame) -> pd.DataFrame:
    """Attach hub attributes and engineered dispatch-time features without reading delivery_hours."""
    out = clean.merge(
        hubs[["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]],
        on="hub_id", how="left", validate="many_to_one",
    )
    out["promised_hours"] = out["service_type"].map(PROMISED_HOURS).astype("int64")
    out["is_cod"] = out["payment_mode"].eq("COD").astype("int64")
    out["is_weekend"] = out["dispatch_time"].dt.dayofweek.ge(5).astype("int64")
    out["dispatch_hour"] = out["dispatch_time"].dt.hour.astype("int64")
    out["log_declared_value"] = np.log(out["declared_value_inr"].astype(float))
    return out

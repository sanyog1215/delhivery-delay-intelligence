"""Part A: data quality, modelling table, delay evidence and correlations."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.shipment_prep import PROMISED_HOURS, FEATURES, clean_shipments, build_features

ROOT = Path(__file__).resolve().parent
DATA, OUT = ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True)

raw = pd.read_csv(DATA / "raw_shipments.csv")
hubs = pd.read_csv(DATA / "hubs.csv")
raw_rows = len(raw)
dedup = raw.drop_duplicates().copy()
spellings_before = int(dedup["service_type"].nunique())
missing_distance = int(dedup["distance_km"].isna().sum())
missing_weight = int(dedup["parcel_weight_kg"].isna().sum())
sentinel_count = int((dedup["parcel_weight_kg"] > 100).sum())
missing_value = int(dedup["declared_value"].isna().sum())
missing_payment = int(dedup["payment_mode"].isna().sum())
clean, medians = clean_shipments(raw)
spellings_after = int(clean["service_type"].nunique())
features = build_features(clean, hubs)
features["is_delayed"] = (features["delivery_hours"] > features["promised_hours"]).astype(int)
features.to_csv(DATA / "modelling_table.csv", index=False)

filled_values = features["declared_value_inr"]
log_values = np.log(filled_values)
hours = features["delivery_hours"].astype(float)
q1, q3 = hours.quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
checks = [
    ("raw_rows", raw_rows),
    ("duplicate_rows_removed", raw_rows - len(dedup)),
    ("rows_after_dedup", len(dedup)),
    ("missing_distance_km_before", missing_distance),
    ("missing_parcel_weight_before", missing_weight),
    ("sentinel_weight_rows", sentinel_count),
    ("missing_declared_value_before", missing_value),
    ("missing_payment_mode_before", missing_payment),
    ("service_type_spellings_before", spellings_before),
    ("service_type_spellings_after", spellings_after),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("rows_after_hub_merge", len(features)),
    ("nulls_in_features_and_target", int(features[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum())),
    ("delayed_count", int(features["is_delayed"].sum())),
    ("delay_rate", float(features["is_delayed"].mean())),
    ("hours_q1", q1), ("hours_q3", q3),
    ("hours_lower_bound", lower), ("hours_upper_bound", upper),
    ("hours_outlier_count", int(((hours < lower) | (hours > upper)).sum())),
    ("declared_value_mean_inr", filled_values.mean()),
    ("declared_value_median_inr", filled_values.median()),
    ("log_declared_value_mean", log_values.mean()),
    ("log_declared_value_median", log_values.median()),
]
pd.DataFrame(checks, columns=["metric", "value"]).round(4).to_csv(OUT / "part_a_checks.csv", index=False, float_format="%.4f")

pieces = []
for dim in ["service_type", "hub_tier"]:
    grp = features.groupby(dim, dropna=False).agg(
        shipments=("shipment_id", "size"), delayed=("is_delayed", "sum"), mean_hours=("delivery_hours", "mean")
    ).reset_index().rename(columns={dim: "group"})
    grp.insert(0, "dimension", dim)
    pieces.append(grp[["dimension", "group", "shipments", "delayed", "mean_hours"]])

features["distance_band"] = pd.cut(
    features["distance_km"], bins=[0, 200, 400, 600, 2000], labels=["0-200", "201-400", "401-600", "601+"],
    right=True, include_lowest=True,
)
grp = features.groupby("distance_band", observed=False).agg(
    shipments=("shipment_id", "size"), delayed=("is_delayed", "sum"), mean_hours=("delivery_hours", "mean")
).reset_index().rename(columns={"distance_band": "group"})
grp.insert(0, "dimension", "distance_band")
pieces.append(grp[["dimension", "group", "shipments", "delayed", "mean_hours"]])
by_group = pd.concat(pieces, ignore_index=True)
by_group["delay_rate"] = by_group["delayed"] / by_group["shipments"]
by_group = by_group[["dimension", "group", "shipments", "delayed", "delay_rate", "mean_hours"]]
by_group.round(4).to_csv(OUT / "part_a_delay_by_group.csv", index=False, float_format="%.4f")

numeric = features.select_dtypes(include=[np.number]).corr(numeric_only=True)
numeric.to_csv(OUT / "part_a_correlations.csv", index=True, index_label="column", float_format="%.4f")
print(f"Part A complete: {len(features)} rows; delayed={features['is_delayed'].sum()}; outputs written.")

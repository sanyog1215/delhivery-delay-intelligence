"""Part D: profile and segment hubs, quantify cluster separation and tier mismatches."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from src.shipment_prep import FEATURES

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
df = pd.read_csv(ROOT / "data" / "modelling_table.csv")
profile = df.groupby("hub_id", as_index=False).agg(
    shipments=("shipment_id", "count"),
    mean_delivery_hours=("delivery_hours", "mean"),
    delay_rate=("is_delayed", "mean"),
    mean_distance_km=("distance_km", "mean"),
    mean_n_stops=("n_stops", "mean"),
).sort_values("hub_id").reset_index(drop=True)
profile = profile.round(4)
profile.to_csv(OUT / "hub_profile.csv", index=False, float_format="%.4f")

PROFILE_COLS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
# Use the rounded, persisted profile values, in the pinned ascending hub_id order.
profile = pd.read_csv(OUT / "hub_profile.csv").sort_values("hub_id").reset_index(drop=True)
scaled = StandardScaler().fit_transform(profile[PROFILE_COLS])

# Separation ratio: mean across-cluster pair distance / mean within-cluster pair distance.
pair_i, pair_j = np.triu_indices(len(profile), k=1)
dists = np.linalg.norm(scaled[pair_i] - scaled[pair_j], axis=1)
sep_rows = []
for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(scaled)
    same = labels[pair_i] == labels[pair_j]
    across_mean = float(dists[~same].mean())
    within_mean = float(dists[same].mean())
    sizes = sorted(np.bincount(labels).tolist(), reverse=True)
    sep_rows.append({"K": k, "across_over_within": across_mean / within_mean,
                     "cluster_sizes": "-".join(map(str, sizes))})
pd.DataFrame(sep_rows).round(4).to_csv(OUT / "part_d_separation.csv", index=False, float_format="%.4f")

km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = km3.fit_predict(scaled)
profile["cluster"] = labels
cluster_delay = profile.groupby("cluster")["delay_rate"].mean().sort_values()
cluster_names = {int(cluster_delay.index[0]): "Standard", int(cluster_delay.index[1]): "Watch", int(cluster_delay.index[2]): "Escalate"}
profile["segment"] = profile["cluster"].map(cluster_names)
hubs = pd.read_csv(ROOT / "data" / "hubs.csv")
segments = profile[["hub_id", "segment"]].merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left", validate="one_to_one")
segments = segments[["hub_id", "segment", "hub_city", "hub_tier"]].sort_values("hub_id")
segments.to_csv(OUT / "hub_segments.csv", index=False)

# Profile aggregates are means of hub-level profile values, as required.
segment_profile = profile.groupby("segment", as_index=False).agg(
    hubs=("hub_id", "count"),
    mean_delivery_hours=("mean_delivery_hours", "mean"),
    delay_rate=("delay_rate", "mean"),
    mean_distance_km=("mean_distance_km", "mean"),
    mean_n_stops=("mean_n_stops", "mean"),
)
segment_order = {"Standard": 0, "Watch": 1, "Escalate": 2}
segment_profile["_order"] = segment_profile["segment"].map(segment_order)
segment_profile = segment_profile.sort_values("_order").drop(columns="_order")
segment_profile.round(4).to_csv(OUT / "segment_profile.csv", index=False, float_format="%.4f")
expected_segment = {1: "Standard", 2: "Watch", 3: "Escalate"}
mismatch = segments[segments.apply(lambda r: r["segment"] != expected_segment.get(int(r["hub_tier"])), axis=1)]
mismatch.to_csv(OUT / "tier_vs_segment_mismatch.csv", index=False)
print(f"Part D complete: {len(profile)} hubs; segments={segment_profile[['segment','hubs']].to_dict('records')}; mismatches={len(mismatch)}")

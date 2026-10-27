"""Score the 48 new shipments and produce the operational action list."""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from src.shipment_prep import clean_shipments, build_features

ROOT = Path(__file__).resolve().parent
DATA, OUT, MODELS = ROOT / "data", ROOT / "outputs", ROOT / "models"
raw_new = pd.read_csv(DATA / "new_shipments.csv")
hubs = pd.read_csv(DATA / "hubs.csv")
medians_df = pd.read_csv(MODELS / "impute_values.csv")
medians = {row["column"]: float(row["median"]) for _, row in medians_df.iterrows()}
threshold = float(pd.read_csv(MODELS / "delay_threshold.csv")["threshold"].iloc[0])
feature_order = pd.read_csv(MODELS / "feature_order.csv")["feature"].tolist()
segments = pd.read_csv(OUT / "hub_segments.csv")[["hub_id", "segment"]]
model = joblib.load(MODELS / "delay_model.pkl")
clean, _ = clean_shipments(raw_new, medians=medians)
featured = build_features(clean, hubs)
X_new = featured[feature_order]
probabilities = model.predict_proba(X_new)[:, 1]
flagged = (probabilities >= threshold).astype(int)
actions = featured[["shipment_id", "hub_id"]].copy()
actions["delay_probability"] = np.round(probabilities, 4)
actions["flagged"] = flagged
actions = actions.merge(segments, on="hub_id", how="left", validate="many_to_one")
actions["action"] = np.where(
    actions["flagged"].eq(0), "No action",
    np.where(actions["segment"].eq("Escalate"), "Escalate to hub manager", "Proactive customer call")
)
actions = actions[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
actions.to_csv(OUT / "new_shipment_actions.csv", index=False, float_format="%.4f")
print(f"Scored {len(actions)} new parcels; flagged={int(actions['flagged'].sum())}; hub escalations={(actions['action'] == 'Escalate to hub manager').sum()}")

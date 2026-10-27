"""Rebuild Part C's final model and persist the reproducible inference artefacts."""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from src.shipment_prep import FEATURES, clean_shipments

ROOT = Path(__file__).resolve().parent
DATA, OUT, MODELS = ROOT / "data", ROOT / "outputs", ROOT / "models"
MODELS.mkdir(exist_ok=True)
df = pd.read_csv(DATA / "modelling_table.csv")
X, y = df[FEATURES], df["is_delayed"].astype(int)
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
config = pd.read_csv(OUT / "part_c_final_config.csv").iloc[0]
max_depth = None if str(config["max_depth"]).lower() == "unlimited" else int(config["max_depth"])
params = {
    "n_estimators": int(config["n_estimators"]),
    "max_depth": max_depth,
    "min_samples_leaf": int(config["min_samples_leaf"]),
}
model = Pipeline([("selector", SelectKBest(f_classif, k=int(config["selector_k"]))),
                  ("model", RandomForestClassifier(random_state=42, **params))])
model.fit(X_train, y_train)
threshold = float(config["threshold"])
joblib.dump(model, MODELS / "delay_model.pkl")
pd.DataFrame({"threshold": [threshold]}).to_csv(MODELS / "delay_threshold.csv", index=False, float_format="%.4f")
pd.DataFrame({"feature": FEATURES}).to_csv(MODELS / "feature_order.csv", index=False)
raw = pd.read_csv(DATA / "raw_shipments.csv")
_, medians = clean_shipments(raw)
pd.DataFrame([{"column": k, "median": v} for k, v in medians.items()]).to_csv(MODELS / "impute_values.csv", index=False, float_format="%.4f")
reloaded = joblib.load(MODELS / "delay_model.pkl")
prob_in_memory = model.predict_proba(X_test)[:, 1]
prob_reloaded = reloaded.predict_proba(X_test)[:, 1]
reload_diff = float(np.max(np.abs(prob_in_memory - prob_reloaded)))
pd.DataFrame([
    {"metric": "test_rows_scored", "value": len(X_test)},
    {"metric": "reload_max_abs_probability_diff", "value": reload_diff},
    {"metric": "threshold", "value": threshold},
]).round(4).to_csv(OUT / "part_d_reload_check.csv", index=False, float_format="%.4f")
print(f"Model packaged: {len(X_test)} test rows; reload max abs diff={reload_diff:.12g}; threshold={threshold:.1f}")

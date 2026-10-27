"""Part C: leakage-aware delay classifier, feature selection, tuning, threshold and test."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.shipment_prep import FEATURES, NOISE_COLUMNS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
df = pd.read_csv(ROOT / "data" / "modelling_table.csv")
X, y = df[FEATURES], df["is_delayed"].astype(int)
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)


def classification_metrics(y_true, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0)}

# Baseline and three validated classifier families.
model_rows = []
model_rows.append({"model": "blind_all_on_time", **classification_metrics(y_val, np.zeros(len(y_val), dtype=int)), "precision": 0.0})
logreg = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000))])
logreg.fit(X_train, y_train)
model_rows.append({"model": "logistic_regression", **classification_metrics(y_val, logreg.predict(X_val))})

depth_candidates = []
for depth in [2, 3, 4, 5, 6, 8, 10]:
    model = DecisionTreeClassifier(criterion="gini", max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    m = classification_metrics(y_val, model.predict(X_val))
    depth_candidates.append((depth, m["f1"], m))
best_tree_depth, _, best_tree_metrics = sorted(depth_candidates, key=lambda x: (-x[1], x[0]))[0]
model_rows.append({"model": f"decision_tree_depth_{best_tree_depth}", **best_tree_metrics})
rf_default = RandomForestClassifier(n_estimators=100, random_state=42)
rf_default.fit(X_train, y_train)
model_rows.append({"model": "random_forest_default", **classification_metrics(y_val, rf_default.predict(X_val))})
pd.DataFrame(model_rows, columns=["model", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"]).round(4).to_csv(OUT / "part_c_models.csv", index=False, float_format="%.4f")

# Signal audit: full 17-feature forest importance and univariate training-only F scores.
rf_signal = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
selector_all = SelectKBest(f_classif, k="all").fit(X_train, y_train)
f_scores = np.nan_to_num(selector_all.scores_, nan=0.0, posinf=0.0, neginf=0.0)
score_frame = pd.DataFrame({"feature": FEATURES, "f_score": f_scores,
                            "rf_importance": rf_signal.feature_importances_,
                            "is_noise": [f in NOISE_COLUMNS for f in FEATURES]})
score_frame["f_rank"] = score_frame["f_score"].rank(method="min", ascending=False).astype(int)
score_frame = score_frame[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
score_frame.round(4).to_csv(OUT / "part_c_feature_scores.csv", index=False, float_format="%.4f")

selectk_rows = []
for k in [5, 7, 10, 12, 17]:
    pipe = Pipeline([("selector", SelectKBest(f_classif, k=k)),
                     ("model", RandomForestClassifier(n_estimators=100, random_state=42))])
    pipe.fit(X_train, y_train)
    selectk_rows.append({"k": k, "val_f1": f1_score(y_val, pipe.predict(X_val), zero_division=0)})
pd.DataFrame(selectk_rows).round(4).to_csv(OUT / "part_c_selectk.csv", index=False, float_format="%.4f")

# Fit the selector on training data only; do all tuning on the selected 7 columns.
selector = SelectKBest(f_classif, k=7).fit(X_train, y_train)
X_train_7, X_val_7 = selector.transform(X_train), selector.transform(X_val)
param_grid = {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5, 10]}
base = RandomForestClassifier(random_state=42)
grid = GridSearchCV(base, param_grid=param_grid, cv=5)
grid.fit(X_train_7, y_train)
rand = RandomizedSearchCV(base, param_distributions=param_grid, n_iter=9, cv=5, random_state=42)
rand.fit(X_train_7, y_train)
default7 = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=None, min_samples_leaf=1).fit(X_train_7, y_train)

def params_text(params):
    return json.dumps(params, sort_keys=True, separators=(", ", ": "))

tuning_rows = [
    {"search": "grid_search", "candidates": 27, "cv_folds": 5, "total_fits": 135,
     "best_params": params_text(grid.best_params_), "best_cv_accuracy": grid.best_score_,
     "val_f1": f1_score(y_val, grid.best_estimator_.predict(X_val_7), zero_division=0)},
    {"search": "randomized_search", "candidates": 9, "cv_folds": 5, "total_fits": 45,
     "best_params": params_text(rand.best_params_), "best_cv_accuracy": rand.best_score_,
     "val_f1": f1_score(y_val, rand.best_estimator_.predict(X_val_7), zero_division=0)},
    {"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1,
     "best_params": "defaults", "best_cv_accuracy": np.nan,
     "val_f1": f1_score(y_val, default7.predict(X_val_7), zero_division=0)},
]
pd.DataFrame(tuning_rows, columns=["search", "candidates", "cv_folds", "total_fits", "best_params", "best_cv_accuracy", "val_f1"]).round(4).to_csv(OUT / "part_c_tuning.csv", index=False, float_format="%.4f")

# Deterministic tie precedence: default forest, then grid, then randomized.
priority = {"default_forest_same_7_features": 0, "grid_search": 1, "randomized_search": 2}
chosen_row = sorted(tuning_rows, key=lambda r: (-r["val_f1"], priority[r["search"]]))[0]
if chosen_row["search"] == "default_forest_same_7_features":
    chosen_params = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}
elif chosen_row["search"] == "grid_search":
    chosen_params = grid.best_params_
else:
    chosen_params = rand.best_params_

final_model = Pipeline([("selector", SelectKBest(f_classif, k=7)),
                        ("model", RandomForestClassifier(random_state=42, **chosen_params))])
final_model.fit(X_train, y_train)
val_prob = final_model.predict_proba(X_val)[:, 1]
threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
threshold_rows = []
for threshold in threshold_values:
    pred = (val_prob >= threshold).astype(int)
    threshold_rows.append({"threshold": threshold, **classification_metrics(y_val, pred)})
pd.DataFrame(threshold_rows, columns=["threshold", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"]).round(4).to_csv(OUT / "part_c_thresholds.csv", index=False, float_format="%.4f")
valid_thresholds = [r["threshold"] for r in threshold_rows if r["recall"] >= 0.80]
chosen_threshold = max(valid_thresholds) if valid_thresholds else 0.1

# Single final test evaluation, after all model/threshold decisions are fixed.
test_prob = final_model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= chosen_threshold).astype(int)
test_metrics = classification_metrics(y_test, test_pred)
pd.DataFrame([{ "threshold": chosen_threshold, **test_metrics }], columns=["threshold", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"]).round(4).to_csv(OUT / "part_c_final_test.csv", index=False, float_format="%.4f")
max_depth_value = chosen_params.get("max_depth")
pd.DataFrame([{
    "selector_k": 7, "n_estimators": chosen_params.get("n_estimators", 100),
    "max_depth": "unlimited" if max_depth_value is None else max_depth_value,
    "min_samples_leaf": chosen_params.get("min_samples_leaf", 1),
    "threshold": chosen_threshold, "chosen_from": chosen_row["search"],
}]).to_csv(OUT / "part_c_final_config.csv", index=False, float_format="%.4f")
print(f"Part C complete: train/val/test delayed={int(y_train.sum())}/{int(y_val.sum())}/{int(y_test.sum())}; chosen={chosen_row['search']}; threshold={chosen_threshold:.1f}; test F1={test_metrics['f1']:.4f}")

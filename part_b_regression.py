"""Part B: delivery-hours regression comparisons and final test evaluation."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
df = pd.read_csv(ROOT / "data" / "modelling_table.csv")
X, y = df[FEATURES], df["delivery_hours"]
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)
checks = [
    ("train_rows", len(X_train)), ("val_rows", len(X_val)), ("test_rows", len(X_test)),
    ("train_scaled_max_abs_mean", float(np.abs(X_train_scaled.mean(axis=0)).max())),
    ("test_scaled_max_abs_mean", float(np.abs(X_test_scaled.mean(axis=0)).max())),
]
pd.DataFrame(checks, columns=["metric", "value"]).round(4).to_csv(OUT / "part_b_checks.csv", index=False, float_format="%.4f")

def metrics(y_true, pred):
    mse = mean_squared_error(y_true, pred)
    return {"MAE": mean_absolute_error(y_true, pred), "MSE": mse, "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, pred)}

# Depth diagnostic: same training rows, independent models.
depths = [1, 2, 3, 4, 6, 8, 12, None]
depth_rows = []
depth_models = {}
for depth in depths:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    depth_models[depth] = model
    depth_rows.append({"max_depth": "unlimited" if depth is None else depth,
                       "train_rmse": np.sqrt(mean_squared_error(y_train, model.predict(X_train))),
                       "val_rmse": np.sqrt(mean_squared_error(y_val, model.predict(X_val)))})
pd.DataFrame(depth_rows).round(4).to_csv(OUT / "part_b_tree_depth.csv", index=False, float_format="%.4f")
best_depth = min(depth_rows, key=lambda r: (r["val_rmse"], float("inf") if r["max_depth"] == "unlimited" else r["max_depth"]))["max_depth"]
best_depth = None if best_depth == "unlimited" else int(best_depth)

models = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}
metric_rows = []
val_results = {}
mean_pred = np.full(len(y_val), float(y_train.mean()))
val_results["mean_baseline"] = metrics(y_val, mean_pred)
metric_rows.append({"model": "mean_baseline", "split": "val", **val_results["mean_baseline"]})
for name, model in models.items():
    model.fit(X_train, y_train)
    res = metrics(y_val, model.predict(X_val))
    val_results[name] = res
    metric_rows.append({"model": name, "split": "val", **res})

lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    lr_rows.append({"learning_rate": lr, "val_rmse": np.sqrt(mean_squared_error(y_val, pred)), "val_r2": r2_score(y_val, pred)})
pd.DataFrame(lr_rows).round(4).to_csv(OUT / "part_b_gb_learning_rate.csv", index=False, float_format="%.4f")

# Compare the two best validation candidates on the combined train + validation set.
ranked = sorted(models, key=lambda n: (val_results[n]["RMSE"], n))
X_tv = pd.concat([X_train, X_val], axis=0)
y_tv = pd.concat([y_train, y_val], axis=0)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []
for name in ranked[:2]:
    estimator = models[name]
    scores = cross_val_score(estimator, X_tv, y_tv, cv=cv)  # default R2 scoring
    cv_rows.append({"model": name, "mean_r2": scores.mean(), "std_r2": scores.std()})
pd.DataFrame(cv_rows).round(4).to_csv(OUT / "part_b_cv.csv", index=False, float_format="%.4f")

winner = min(models, key=lambda n: (val_results[n]["RMSE"], n))
test_result = metrics(y_test, models[winner].predict(X_test))
metric_rows.append({"model": winner, "split": "test", **test_result})
pd.DataFrame(metric_rows, columns=["model", "split", "MAE", "MSE", "RMSE", "R2"]).round(4).to_csv(OUT / "part_b_metrics.csv", index=False, float_format="%.4f")
print(f"Part B complete: validation winner={winner}, RMSE={val_results[winner]['RMSE']:.4f}, test R2={test_result['R2']:.4f}")

"""Part B - how many hours will a parcel actually take?

Two-step split (70/15/15), a scaler fitted on training rows only, a baseline,
a tree-depth diagnostic, four candidate regressors, a learning-rate experiment,
a 5-fold cross-check on train+validation, and exactly one test evaluation of the
winning model.

Run from the repository root:  python part_b_regression.py
Reads : data/modelling_table.csv
Writes: outputs/part_b_checks.csv, part_b_tree_depth.csv, part_b_metrics.csv,
        part_b_gb_learning_rate.csv, part_b_cv.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from shipment_prep import FEATURES  # noqa: E402

DATA_PATH = os.path.join("data", "modelling_table.csv")
OUT_DIR = "outputs"
DEPTHS = [1, 2, 3, 4, 6, 8, 12, None]
LEARNING_RATES = [0.01, 0.1, 1.0]


def format_metric(value):
    """Counts stay integers; measures are rounded to 4 decimals and trimmed."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def write_csv(df, name):
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(4)
    out.to_csv(os.path.join(OUT_DIR, name), index=False)
    return out


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }


def depth_label(depth):
    return "unlimited" if depth is None else depth


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df["delivery_hours"]

    # ------------------------------------------------ 1. two-step 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # ------------------------------------- 2. scaling fitted on train only
    scaler = StandardScaler().fit(X_train)
    train_scaled = scaler.transform(X_train)
    test_scaled = scaler.transform(X_test)
    checks = pd.DataFrame(
        [
            ("train_rows", len(X_train)),
            ("val_rows", len(X_val)),
            ("test_rows", len(X_test)),
            ("train_scaled_max_abs_mean", float(np.abs(train_scaled.mean(axis=0)).max())),
            ("test_scaled_max_abs_mean", float(np.abs(test_scaled.mean(axis=0)).max())),
        ],
        columns=["metric", "value"],
    )
    checks["value"] = [format_metric(v) for v in checks["value"]]
    checks.to_csv(os.path.join(OUT_DIR, "part_b_checks.csv"), index=False)

    # -------------------------------------------------------- 3. baseline
    baseline_pred = np.full(len(y_val), y_train.mean())
    rows = [{"model": "mean_baseline", "split": "val", **regression_metrics(y_val, baseline_pred)}]

    # ------------------------------------------ 4. tree-depth diagnostic
    depth_rows = []
    for depth in DEPTHS:
        tree = DecisionTreeRegressor(max_depth=depth, random_state=42).fit(X_train, y_train)
        depth_rows.append(
            {
                "max_depth": depth_label(depth),
                "train_rmse": float(np.sqrt(mean_squared_error(y_train, tree.predict(X_train)))),
                "val_rmse": float(np.sqrt(mean_squared_error(y_val, tree.predict(X_val)))),
            }
        )
    depth_df = pd.DataFrame(depth_rows)
    write_csv(depth_df, "part_b_tree_depth.csv")

    # Ties are broken towards the shallower tree: the list is already ascending in depth.
    best_idx = depth_df["val_rmse"].idxmin()
    best_depth = DEPTHS[best_idx]
    print(f"best tree depth: {depth_label(best_depth)} (val RMSE {depth_df.loc[best_idx, 'val_rmse']:.4f})")

    # -------------------------------------------- 5. four candidate models
    candidates = {
        "linear_regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42
        ),
    }
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        rows.append({"model": name, "split": "val", **regression_metrics(y_val, model.predict(X_val))})

    metrics_df = pd.DataFrame(rows)[["model", "split", "MAE", "MSE", "RMSE", "R2"]]

    # -------------------------------------- 6. gradient-boosting learning rate
    lr_rows = []
    for lr in LEARNING_RATES:
        gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
        gb.fit(X_train, y_train)
        pred = gb.predict(X_val)
        lr_rows.append(
            {
                "learning_rate": lr,
                "val_rmse": float(np.sqrt(mean_squared_error(y_val, pred))),
                "val_r2": r2_score(y_val, pred),
            }
        )
    write_csv(pd.DataFrame(lr_rows), "part_b_gb_learning_rate.csv")

    # ------------------------------- 7. cross-validation of the two best models
    ranked = metrics_df[metrics_df["split"] == "val"].sort_values("RMSE")
    top_two = ranked["model"].iloc[:2].tolist()
    final_name = ranked["model"].iloc[0]

    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # one object, both models
    cv_rows = []
    for name in top_two:
        scores = cross_val_score(candidates[name], X_trainval, y_trainval, cv=kfold)
        cv_rows.append({"model": name, "mean_r2": float(scores.mean()), "std_r2": float(scores.std())})
    cv_df = write_csv(pd.DataFrame(cv_rows), "part_b_cv.csv")

    # One test evaluation, for the winner only.
    final_model = candidates[final_name]
    metrics_df = pd.concat(
        [
            metrics_df,
            pd.DataFrame(
                [{"model": final_name, "split": "test",
                  **regression_metrics(y_test, final_model.predict(X_test))}]
            ),
        ],
        ignore_index=True,
    )
    metrics_df = write_csv(metrics_df, "part_b_metrics.csv")

    print(f"validation ranking: {top_two} | final model: {final_name}")
    print(metrics_df.to_string(index=False))
    print(cv_df.to_string(index=False))
    print(depth_df.to_string(index=False))


if __name__ == "__main__":
    main()

"""Package the Part C classifier as a reusable artefact, then prove the pickle round-trips.

Run from the repository root (after part_c_classification.py):
    python package_model.py

Reads : data/modelling_table.csv, data/raw_shipments.csv, outputs/part_c_final_config.csv
Writes: models/delay_model.pkl, models/delay_threshold.csv, models/feature_order.csv,
        models/impute_values.csv, outputs/part_d_reload_check.csv
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from shipment_prep import FEATURES, clean_shipments  # noqa: E402

MODEL_DIR = "models"
OUT_DIR = "outputs"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join("data", "modelling_table.csv"))
    config = pd.read_csv(os.path.join(OUT_DIR, "part_c_final_config.csv")).iloc[0]

    X = df[FEATURES]
    y = df["is_delayed"]

    # Rebuild exactly the Part C split: same two calls, same seeds, stratified.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    max_depth = None if str(config["max_depth"]) == "unlimited" else int(config["max_depth"])
    pipeline = Pipeline(
        [
            ("selector", SelectKBest(score_func=f_classif, k=int(config["selector_k"]))),
            (
                "model",
                RandomForestClassifier(
                    random_state=42,
                    n_estimators=int(config["n_estimators"]),
                    max_depth=max_depth,
                    min_samples_leaf=int(config["min_samples_leaf"]),
                ),
            ),
        ]
    ).fit(X_train, y_train)

    threshold = float(config["threshold"])
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "delay_model.pkl"))

    pd.DataFrame({"threshold": [threshold]}).to_csv(
        os.path.join(MODEL_DIR, "delay_threshold.csv"), index=False
    )
    pd.DataFrame({"feature": FEATURES}).to_csv(
        os.path.join(MODEL_DIR, "feature_order.csv"), index=False
    )

    _, medians = clean_shipments(pd.read_csv(os.path.join("data", "raw_shipments.csv")))
    pd.DataFrame(
        {
            "column": list(medians.keys()),
            "median": [round(float(v), 4) for v in medians.values()],
        }
    ).to_csv(os.path.join(MODEL_DIR, "impute_values.csv"), index=False)

    # -------------------------------------------------- reload verification
    reloaded = joblib.load(os.path.join(MODEL_DIR, "delay_model.pkl"))
    in_memory = pipeline.predict_proba(X_test)[:, 1]
    from_disk = reloaded.predict_proba(X_test)[:, 1]
    max_diff = float(np.max(np.abs(in_memory - from_disk)))

    pd.DataFrame(
        [
            ["test_rows_scored", str(len(X_test))],
            ["reload_max_abs_probability_diff", f"{max_diff:.10f}".rstrip("0").rstrip(".") or "0"],
            ["threshold", f"{threshold:.4f}".rstrip("0").rstrip(".")],
        ],
        columns=["metric", "value"],
    ).to_csv(os.path.join(OUT_DIR, "part_d_reload_check.csv"), index=False)

    print(f"saved model with {config['chosen_from']}; test rows {len(X_test)}; max diff {max_diff:.2e}")


if __name__ == "__main__":
    main()

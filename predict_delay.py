"""Turn the 48 parcels about to be dispatched into an operations action list.

The cleaning and feature code is imported from src/shipment_prep.py - the same
functions that built the training table - so training and scoring cannot drift
apart. Medians come from models/impute_values.csv, never recomputed on new data.

Run from the repository root (after part_d_segments.py and package_model.py):
    python predict_delay.py
Reads : data/new_shipments.csv, data/hubs.csv, models/*, outputs/hub_segments.csv
Writes: outputs/new_shipment_actions.csv
"""
import os
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from shipment_prep import build_features, clean_shipments  # noqa: E402

OUT_PATH = os.path.join("outputs", "new_shipment_actions.csv")


def main():
    new_raw = pd.read_csv(os.path.join("data", "new_shipments.csv"))
    hubs = pd.read_csv(os.path.join("data", "hubs.csv"))

    model = joblib.load(os.path.join("models", "delay_model.pkl"))
    threshold = float(pd.read_csv(os.path.join("models", "delay_threshold.csv"))["threshold"].iloc[0])
    feature_order = pd.read_csv(os.path.join("models", "feature_order.csv"))["feature"].tolist()
    impute = pd.read_csv(os.path.join("models", "impute_values.csv"))
    medians = dict(zip(impute["column"], impute["median"]))
    segments = pd.read_csv(os.path.join("outputs", "hub_segments.csv"))[["hub_id", "segment"]]

    clean, _ = clean_shipments(new_raw, medians=medians)
    featured = build_features(clean, hubs)

    X = featured[feature_order]
    featured["delay_probability"] = model.predict_proba(X)[:, 1].round(4)
    featured["flagged"] = (featured["delay_probability"] >= threshold).astype(int)

    featured = featured.merge(segments, on="hub_id", how="left", validate="many_to_one")
    featured["action"] = "No action"
    flagged_mask = featured["flagged"] == 1
    featured.loc[flagged_mask & (featured["segment"] == "Escalate"), "action"] = "Escalate to hub manager"
    featured.loc[flagged_mask & (featured["segment"] != "Escalate"), "action"] = "Proactive customer call"

    out = featured[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]].copy()
    out["delay_probability"] = out["delay_probability"].round(4)
    out.to_csv(OUT_PATH, index=False)

    print(out["action"].value_counts().to_string())
    print(f"rows written: {len(out)} -> {OUT_PATH}")


if __name__ == "__main__":
    main()

"""Part A - build the modelling table, the audit checks, the delay evidence and correlations.

Run from the repository root:  python part_a_clean_eda.py
Reads : data/raw_shipments.csv, data/hubs.csv
Writes: data/modelling_table.csv, outputs/part_a_checks.csv,
        outputs/part_a_delay_by_group.csv, outputs/part_a_correlations.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from shipment_prep import FEATURES, build_features, clean_shipments  # noqa: E402

DATA_DIR = "data"
OUT_DIR = "outputs"
GROUP_ORDER = ["service_type", "hub_tier", "distance_band"]


def format_metric(value):
    """Counts stay integers; measures are rounded to 4 decimals and trimmed."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def write_csv(df, path):
    """Write with index=False and every float rounded to 4 decimals."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(4)
    out.to_csv(path, index=False)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    raw = pd.read_csv(os.path.join(DATA_DIR, "raw_shipments.csv"))
    hubs = pd.read_csv(os.path.join(DATA_DIR, "hubs.csv"))

    # "Before" counts: after removing duplicates, before any filling.
    deduped = raw.drop_duplicates(keep="first").reset_index(drop=True)

    clean, medians = clean_shipments(raw)
    model_df = build_features(clean, hubs)
    model_df["is_delayed"] = (model_df["delivery_hours"] > model_df["promised_hours"]).astype(int)

    write_csv(model_df, os.path.join(DATA_DIR, "modelling_table.csv"))

    # ---------------------------------------------------------------- checks
    hours = model_df["delivery_hours"]
    q1, q3 = hours.quantile(0.25), hours.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier_count = int(((hours < lower) | (hours > upper)).sum())

    checks = [
        ("raw_rows", int(len(raw))),
        ("duplicate_rows_removed", int(len(raw) - len(deduped))),
        ("rows_after_dedup", int(len(deduped))),
        ("missing_distance_km_before", int(deduped["distance_km"].isna().sum())),
        ("missing_parcel_weight_before", int(deduped["parcel_weight_kg"].isna().sum())),
        ("sentinel_weight_rows", int((deduped["parcel_weight_kg"] > 100).sum())),
        ("missing_declared_value_before", int(deduped["declared_value"].isna().sum())),
        ("missing_payment_mode_before", int(deduped["payment_mode"].isna().sum())),
        ("service_type_spellings_before", int(deduped["service_type"].nunique())),
        ("service_type_spellings_after", int(model_df["service_type"].nunique())),
        ("declared_value_mean_inr", float(model_df["declared_value_inr"].mean())),
        ("declared_value_median_inr", float(model_df["declared_value_inr"].median())),
        ("log_declared_value_mean", float(model_df["log_declared_value"].mean())),
        ("log_declared_value_median", float(model_df["log_declared_value"].median())),
        ("median_distance_km", float(medians["distance_km"])),
        ("median_parcel_weight_kg", float(medians["parcel_weight_kg"])),
        ("median_declared_value_inr", float(medians["declared_value_inr"])),
        ("rows_after_hub_merge", int(len(model_df))),
        ("nulls_in_features_and_target",
         int(model_df[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum())),
        ("delayed_count", int(model_df["is_delayed"].sum())),
        ("delay_rate", float(model_df["is_delayed"].mean())),
        ("hours_q1", float(q1)),
        ("hours_q3", float(q3)),
        ("hours_lower_bound", float(lower)),
        ("hours_upper_bound", float(upper)),
        ("hours_outlier_count", outlier_count),
    ]
    check_df = pd.DataFrame(checks, columns=["metric", "value"])
    check_df["value"] = [format_metric(v) for _, v in checks]
    check_df.to_csv(os.path.join(OUT_DIR, "part_a_checks.csv"), index=False)

    # ------------------------------------------------------- delay evidence
    model_df["distance_band"] = pd.cut(
        model_df["distance_km"],
        bins=[0, 200, 400, 600, 2000],
        labels=["0-200", "201-400", "401-600", "601+"],
        right=True,
        include_lowest=True,
    )

    frames = []
    for dim in GROUP_ORDER:
        grouped = (
            model_df.groupby(dim, observed=True)
            .agg(
                shipments=("shipment_id", "size"),
                delayed=("is_delayed", "sum"),
                delay_rate=("is_delayed", "mean"),
                mean_hours=("delivery_hours", "mean"),
            )
            .reset_index()
            .rename(columns={dim: "group"})
        )
        grouped.insert(0, "dimension", dim)
        grouped["group"] = grouped["group"].astype(str)
        frames.append(grouped.sort_values("group"))

    delay_table = pd.concat(frames, ignore_index=True)
    delay_table["delayed"] = delay_table["delayed"].astype(int)
    delay_table["shipments"] = delay_table["shipments"].astype(int)
    write_csv(delay_table, os.path.join(OUT_DIR, "part_a_delay_by_group.csv"))

    # ---------------------------------------------------------- correlations
    numeric = model_df.select_dtypes(include=[np.number])
    corr = numeric.corr().round(4)
    corr.to_csv(os.path.join(OUT_DIR, "part_a_correlations.csv"), index_label="column")

    # ------------------------------------------------------------- console
    print("rows:", len(model_df), "| medians:", medians, "| delayed:", int(model_df["is_delayed"].sum()))
    print(delay_table.to_string(index=False))
    top = (
        corr["delivery_hours"].drop("delivery_hours").abs().sort_values(ascending=False).head(3)
    )
    print("top correlations with delivery_hours:")
    for name, _ in top.items():
        print(f"  {name}: {corr.loc[name, 'delivery_hours']:.4f}")


if __name__ == "__main__":
    main()

"""Shared constants and preprocessing for the Delhivery delay-intelligence pipeline.

Both functions in this module are used by part_a_clean_eda.py (training data) and
by predict_delay.py (new, undelivered parcels). Keeping a single copy of the
logic is what guarantees that training and scoring see the same transformations.
"""
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

NUMERIC_TO_FILL = ["distance_km", "parcel_weight_kg", "declared_value_inr"]
MAX_PLAUSIBLE_WEIGHT_KG = 100.0


def clean_shipments(raw, medians=None):
    """Clean a raw shipment export and return ``(clean_df, medians)``.

    Steps are applied in the order fixed by the brief:
    drop exact duplicates, normalise service_type, fill payment_mode, parse the
    rupee text column, parse dispatch_time, turn scanner sentinels into missing
    values, then median-fill the three numeric gaps.

    Parameters
    ----------
    raw : pandas.DataFrame
        A raw export (historical or new parcels).
    medians : dict or None
        Fill values learned on the historical table. ``None`` means "learn them
        now from the data left after the first six steps".

    Returns
    -------
    (clean_df, medians) : tuple[pandas.DataFrame, dict]
    """
    df = raw.copy()

    # 1. Exact duplicate rows: an export can repeat a row when a batch is re-sent.
    df = df.drop_duplicates(keep="first").reset_index(drop=True)

    # 2. service_type arrives as "express", "EXPRESS" and "  Express ".
    df["service_type"] = df["service_type"].astype(str).str.strip().str.title()

    # 3. A blank payment mode is itself information: keep it as an explicit level.
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # 4. "₹1,290" -> 1290.0, stored under a numeric name; the text column is dropped.
    #    astype(str) turns NaN into the string "nan"; to_numeric maps it back to NaN.
    value_text = (
        df["declared_value"].astype(str)
        .str.replace("\u20b9", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    df["declared_value_inr"] = pd.to_numeric(value_text, errors="coerce")
    df = df.drop(columns=["declared_value"])

    # 5. dispatch_time is text until it is parsed.
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"], format="%Y-%m-%d %H:%M", errors="coerce")

    # 6. The network accepts up to 40 kg, so 999 kg is a scanner fault, not a parcel.
    sentinel = df["parcel_weight_kg"] > MAX_PLAUSIBLE_WEIGHT_KG
    df.loc[sentinel, "parcel_weight_kg"] = np.nan

    # 7. Median fill. Medians are learned once on the historical table and reused
    #    verbatim at scoring time, so a new batch cannot shift the definition.
    if medians is None:
        medians = {col: float(df[col].median()) for col in NUMERIC_TO_FILL}
    else:
        medians = {col: float(medians[col]) for col in NUMERIC_TO_FILL}
    for col in NUMERIC_TO_FILL:
        df[col] = df[col].fillna(medians[col])

    return df, medians


def build_features(clean, hubs):
    """Attach hub attributes and derive the model columns. Never reads delivery_hours."""
    df = clean.merge(hubs, on="hub_id", how="left", validate="many_to_one")

    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"])

    return df

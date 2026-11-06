"""Part D (segmentation half) - build hub profiles, cluster them, and audit the tier labels.

The 24 hubs enter StandardScaler and KMeans in ascending hub_id order, which is
the order groupby returns and the order hub_profile.csv is written in. That fixed
order plus random_state=42 is what makes the K=4 and K=5 solutions reproducible.

Run from the repository root:  python part_d_segments.py
Reads : data/modelling_table.csv
Writes: outputs/hub_profile.csv, part_d_separation.csv, hub_segments.csv,
        segment_profile.csv, tier_vs_segment_mismatch.csv
"""
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join("data", "modelling_table.csv")
OUT_DIR = "outputs"
PROFILE_COLS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
SEGMENTS = ["Standard", "Watch", "Escalate"]  # lowest delay rate to highest
TIER_TO_SEGMENT = {1: "Standard", 2: "Watch", 3: "Escalate"}


def write_csv(df, name, **kwargs):
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(4)
    out.to_csv(os.path.join(OUT_DIR, name), index=False, **kwargs)
    return out


def pair_ratio(scaled, labels):
    """Mean distance across clusters divided by mean distance within clusters."""
    n = len(scaled)
    across, within = [], []
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(scaled[i] - scaled[j]))
            (across if labels[i] != labels[j] else within).append(dist)
    return float(np.mean(across) / np.mean(within))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    # ------------------------------------------------------- 1. hub profile
    profile = (
        df.groupby("hub_id")
        .agg(
            shipments=("shipment_id", "size"),
            mean_delivery_hours=("delivery_hours", "mean"),
            delay_rate=("is_delayed", "mean"),
            mean_distance_km=("distance_km", "mean"),
            mean_n_stops=("n_stops", "mean"),
        )
        .reset_index()
        .sort_values("hub_id")
        .reset_index(drop=True)
    )
    write_csv(profile, "hub_profile.csv")

    # ------------------------------------------------------- 2. K-Means, K=3
    scaled = StandardScaler().fit_transform(profile[PROFILE_COLS])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(scaled)
    profile["cluster"] = kmeans.labels_

    # ------------------------------------------- 3. separation evidence K=2..5
    sep_rows = []
    for k in [2, 3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaled)
        labels = km.labels_
        sizes = sorted(np.bincount(labels), reverse=True)
        sep_rows.append(
            {
                "K": k,
                "across_over_within": pair_ratio(scaled, labels),
                "cluster_sizes": "-".join(str(s) for s in sizes),
            }
        )
    write_csv(pd.DataFrame(sep_rows), "part_d_separation.csv")

    # -------------------------------- 4. segment names, profiles, tier audit
    cluster_delay = profile.groupby("cluster")["delay_rate"].mean().sort_values()
    cluster_to_segment = {int(c): SEGMENTS[i] for i, c in enumerate(cluster_delay.index)}
    profile["segment"] = profile["cluster"].map(cluster_to_segment)

    hubs = pd.read_csv(os.path.join("data", "hubs.csv"))[["hub_id", "hub_city"]]
    segments = (
        profile[["hub_id", "segment"]]
        .merge(hubs, on="hub_id", how="left")
        .merge(pd.read_csv(os.path.join("data", "hubs.csv"))[["hub_id", "hub_tier"]], on="hub_id")
        .sort_values("hub_id")
        .reset_index(drop=True)
    )
    segments = segments[["hub_id", "segment", "hub_city", "hub_tier"]]
    segments["hub_tier"] = segments["hub_tier"].astype(int)
    write_csv(segments, "hub_segments.csv")

    seg_profile = (
        profile.groupby("segment")
        .agg(
            hubs=("hub_id", "size"),
            mean_delivery_hours=("mean_delivery_hours", "mean"),
            delay_rate=("delay_rate", "mean"),
            mean_distance_km=("mean_distance_km", "mean"),
            mean_n_stops=("mean_n_stops", "mean"),
        )
        .reindex(SEGMENTS)
        .reset_index()
    )
    seg_profile["hubs"] = seg_profile["hubs"].astype(int)
    write_csv(seg_profile, "segment_profile.csv")

    segments["implied_segment"] = segments["hub_tier"].map(TIER_TO_SEGMENT)
    mismatch = segments.loc[
        segments["segment"] != segments["implied_segment"],
        ["hub_id", "segment", "hub_city", "hub_tier"],
    ].reset_index(drop=True)
    write_csv(mismatch, "tier_vs_segment_mismatch.csv")

    print(seg_profile.to_string(index=False))
    print(pd.DataFrame(sep_rows).to_string(index=False))
    print("mismatches:\n", mismatch.to_string(index=False))


if __name__ == "__main__":
    main()

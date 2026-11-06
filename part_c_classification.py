"""Part C - which parcels will miss their promise, and at what alert threshold?

Builds the delay classifier on a stratified split, proves that the five
bookkeeping columns carry no signal, tunes the forest on the 7 real features,
and converts probabilities into an operations decision using the rule
"highest threshold whose validation recall is still at least 0.80".

Run from the repository root:  python part_c_classification.py
Reads : data/modelling_table.csv
Writes: outputs/part_c_models.csv, part_c_feature_scores.csv, part_c_selectk.csv,
        part_c_tuning.csv, part_c_thresholds.csv, part_c_final_test.csv,
        part_c_final_config.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from shipment_prep import FEATURES, NOISE_COLUMNS  # noqa: E402

DATA_PATH = os.path.join("data", "modelling_table.csv")
OUT_DIR = "outputs"
TREE_DEPTHS = [2, 3, 4, 5, 6, 8, 10]
K_VALUES = [5, 7, 10, 12, 17]
K_SELECTED = 7
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # exact decimals, not np.arange
TARGET_RECALL = 0.80
PARAM_GRID = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10],
}


def write_csv(df, name):
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(4)
    out.to_csv(os.path.join(OUT_DIR, name), index=False)
    return out


def confusion_row(name, y_true, y_pred):
    """Confusion cells plus metrics recomputed from those cells, so the row is self-consistent."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "model": name, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df["is_delayed"]

    # ------------------------------------------- 1. stratified 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # ------------------------------- 2. blind baseline and three classifiers
    model_rows = []

    blind = np.zeros(len(y_val), dtype=int)
    model_rows.append(confusion_row("blind_all_on_time", y_val, blind))

    logistic = Pipeline(
        [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]
    ).fit(X_train, y_train)
    model_rows.append(confusion_row("logistic_regression", y_val, logistic.predict(X_val)))

    tree_scores = {}
    for depth in TREE_DEPTHS:
        tree = DecisionTreeClassifier(criterion="gini", max_depth=depth, random_state=42)
        tree.fit(X_train, y_train)
        tree_scores[depth] = f1_score(y_val, tree.predict(X_val))
    best_depth = max(sorted(tree_scores), key=lambda d: (tree_scores[d], -d))
    best_tree = DecisionTreeClassifier(criterion="gini", max_depth=best_depth, random_state=42)
    best_tree.fit(X_train, y_train)
    model_rows.append(
        confusion_row(f"decision_tree_depth_{best_depth}", y_val, best_tree.predict(X_val))
    )

    forest_all = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    model_rows.append(confusion_row("random_forest_default", y_val, forest_all.predict(X_val)))

    write_csv(pd.DataFrame(model_rows), "part_c_models.csv")
    print(f"decision-tree depths -> F1: { {d: round(v, 4) for d, v in tree_scores.items()} }")

    # ------------------------------------------- 3. which columns carry signal?
    selector_full = SelectKBest(score_func=f_classif, k="all").fit(X_train, y_train)
    scores = selector_full.scores_
    importances = forest_all.feature_importances_

    feat_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "f_score": scores,
            "rf_importance": importances,
            "is_noise": [f in NOISE_COLUMNS for f in FEATURES],
        }
    )
    feat_df["f_rank"] = feat_df["f_score"].rank(ascending=False, method="min").astype(int)
    feat_df = feat_df.sort_values("f_rank")[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
    feat_df = write_csv(feat_df, "part_c_feature_scores.csv")

    noise_share = float(feat_df.loc[feat_df["is_noise"], "rf_importance"].sum())
    print(f"noise share of forest importance: {noise_share:.4f}")

    selectk_rows = []
    for k in K_VALUES:
        pipe = Pipeline(
            [
                ("selector", SelectKBest(score_func=f_classif, k=k)),
                ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
            ]
        )
        pipe.fit(X_train, y_train)
        selectk_rows.append({"k": k, "val_f1": f1_score(y_val, pipe.predict(X_val))})
    write_csv(pd.DataFrame(selectk_rows), "part_c_selectk.csv")

    # ------------------------------------------------------------ 4. tuning
    selector7 = SelectKBest(score_func=f_classif, k=K_SELECTED).fit(X_train, y_train)
    X_train_7 = selector7.transform(X_train)
    X_val_7 = selector7.transform(X_val)

    base_forest = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(base_forest, PARAM_GRID, cv=5)
    grid.fit(X_train_7, y_train)

    random_search = RandomizedSearchCV(base_forest, PARAM_GRID, n_iter=9, cv=5, random_state=42)
    random_search.fit(X_train_7, y_train)

    default_forest = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_7, y_train)

    tuning_rows = [
        {
            "search": "grid_search",
            "candidates": 27,
            "cv_folds": 5,
            "total_fits": 135,
            "best_params": str(dict(sorted(grid.best_params_.items()))),
            "best_cv_accuracy": float(grid.best_score_),
            "val_f1": f1_score(y_val, grid.best_estimator_.predict(X_val_7)),
        },
        {
            "search": "randomized_search",
            "candidates": 9,
            "cv_folds": 5,
            "total_fits": 45,
            "best_params": str(dict(sorted(random_search.best_params_.items()))),
            "best_cv_accuracy": float(random_search.best_score_),
            "val_f1": f1_score(y_val, random_search.best_estimator_.predict(X_val_7)),
        },
        {
            "search": "default_forest_same_7_features",
            "candidates": 1,
            "cv_folds": 0,
            "total_fits": 1,
            "best_params": "defaults",
            "best_cv_accuracy": np.nan,
            "val_f1": f1_score(y_val, default_forest.predict(X_val_7)),
        },
    ]
    tuning_df = write_csv(pd.DataFrame(tuning_rows), "part_c_tuning.csv")
    print(tuning_df.to_string(index=False))

    # --------------------------------------------- 5. final configuration
    priority = {"default_forest_same_7_features": 2, "grid_search": 1, "randomized_search": 0}
    chosen_row = max(
        tuning_rows,
        key=lambda r: (round(r["val_f1"], 10), priority[r["search"]]),
    )
    chosen_from = chosen_row["search"]
    if chosen_from == "grid_search":
        params = grid.best_params_
    elif chosen_from == "randomized_search":
        params = random_search.best_params_
    else:
        params = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}

    final_model = Pipeline(
        [
            ("selector", SelectKBest(score_func=f_classif, k=K_SELECTED)),
            (
                "model",
                RandomForestClassifier(
                    random_state=42,
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    min_samples_leaf=params["min_samples_leaf"],
                ),
            ),
        ]
    ).fit(X_train, y_train)

    # ------------------------------------------------- 6. alert threshold
    val_prob = final_model.predict_proba(X_val)[:, 1]
    threshold_rows = []
    for t in THRESHOLDS:
        pred = (val_prob >= t).astype(int)
        threshold_rows.append({"threshold": t, **confusion_row("", y_val, pred)})
    threshold_df = pd.DataFrame(threshold_rows).drop(columns=["model"])
    threshold_df = write_csv(threshold_df, "part_c_thresholds.csv")

    qualifying = threshold_df[threshold_df["recall"] >= TARGET_RECALL]["threshold"]
    threshold = float(qualifying.max()) if len(qualifying) else 0.1
    print(f"chosen threshold: {threshold}")

    # ------------------------------------------- 7. one test evaluation
    test_prob = final_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)
    test_row = confusion_row("final_model", y_test, test_pred)
    test_row.pop("model")
    test_df = pd.DataFrame([{"threshold": threshold, **test_row}])
    write_csv(test_df, "part_c_final_test.csv")

    config_df = pd.DataFrame(
        [
            {
                "selector_k": K_SELECTED,
                "n_estimators": params["n_estimators"],
                "max_depth": "unlimited" if params["max_depth"] is None else params["max_depth"],
                "min_samples_leaf": params["min_samples_leaf"],
                "threshold": threshold,
                "chosen_from": chosen_from,
            }
        ]
    )
    config_df.to_csv(os.path.join(OUT_DIR, "part_c_final_config.csv"), index=False)

    print(threshold_df.to_string(index=False))
    print(test_df.to_string(index=False))


if __name__ == "__main__":
    main()

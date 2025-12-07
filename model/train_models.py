# model/train_models.py

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Directory where encoded feature CSVs and models live
BASE_DIR = "Model_Data"

DATA_CONFIG = {
    "fruit": {
        "X_path": os.path.join(BASE_DIR, "fruit_X.csv"),
        "y_path": os.path.join(BASE_DIR, "fruit_y.csv"),
        "clean_path": "Data_preprocessing/fruit_sales_clean.csv",
    },
    "veg": {
        "X_path": os.path.join(BASE_DIR, "veg_X.csv"),
        "y_path": os.path.join(BASE_DIR, "veg_y.csv"),
        "clean_path": "Data_preprocessing/vegetable_sales_clean.csv",
    },
}


def load_dataset(kind: str):
    """
    Load prepared features (X), target (y), and clean dataframe (for date + item).
    X and y are already encoded + scaled in previous preprocessing steps.
    """
    cfg = DATA_CONFIG[kind]

    X = pd.read_csv(cfg["X_path"])
    y = pd.read_csv(cfg["y_path"], header=None).iloc[:, 0]
    df_clean = pd.read_csv(cfg["clean_path"])
    df_clean["date"] = pd.to_datetime(df_clean["date"])

    assert len(X) == len(y) == len(df_clean), "Length mismatch between X, y, clean data"

    # Attach item + date for time-aware split / per-item training
    X["item"] = df_clean["item"].values
    X["date"] = df_clean["date"].values

    return X, y, df_clean


# --------------------------------------------------------------------
# 1) FRUITS → FINAL MODEL = ITEM-WISE LINEAR REGRESSION
# --------------------------------------------------------------------

def train_fruit_itemwise_linear_regression(X: pd.DataFrame, y: pd.Series):
    """
    Train a separate Linear Regression model for each fruit item.
    We still do an 80/20 time-based split inside each item to report MAE/R².
    The trained models (one per item) are saved in a single dict via joblib.
    """
    print("\n===== 🍎 Training FINAL models for FRUITS: Item-wise Linear Regression =====")

    models = {}
    metrics = []

    for item in sorted(X["item"].unique()):
        X_item = X[X["item"] == item].copy().sort_values("date")
        y_item = y.loc[X_item.index]

        X_features = X_item.drop(columns=["item", "date"])

        n = len(X_features)
        if n < 20:
            print(f"  ⚠ Skipping {item}: not enough data ({n} rows).")
            continue

        split_idx = int(0.8 * n)
        X_train, X_test = X_features.iloc[:split_idx], X_features.iloc[split_idx:]
        y_train, y_test = y_item.iloc[:split_idx], y_item.iloc[split_idx:]

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics.append(
            {
                "kind": "fruit",
                "item": item,
                "model": "Final_Itemwise_LinearRegression",
                "mae": mae,
                "r2": r2,
                "n_train": len(X_train),
                "n_test": len(X_test),
            }
        )

        models[item] = model

        print(f"  {item} → MAE={mae:.3f}, R²={r2:.3f} (train={len(X_train)}, test={len(X_test)})")

    # Save all item-wise models into a single file
    model_dir = os.path.join(BASE_DIR, "final_models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "fruit_itemwise_linear_regression.pkl")
    joblib.dump(models, model_path)
    print(f"\n💾 Saved FINAL fruit item-wise models → {model_path}")

    return pd.DataFrame(metrics)


# --------------------------------------------------------------------
# 2) VEGETABLES → FINAL MODEL = GLOBAL RANDOM FOREST
# --------------------------------------------------------------------

def train_veg_global_random_forest(X: pd.DataFrame, y: pd.Series):
    """
    Train ONE global Random Forest model for all vegetables.
    Use time-based 80/20 train-test split for metric reporting.
    """
    print("\n===== 🥦 Training FINAL model for VEGETABLES: Global Random Forest =====")

    # Sort by date for time-aware split
    X_sorted = X.sort_values("date")
    y_sorted = y.loc[X_sorted.index]

    # Drop helper columns from features
    X_features = X_sorted.drop(columns=["item", "date"])

    n = len(X_features)
    split_idx = int(0.8 * n)

    X_train, X_test = X_features.iloc[:split_idx], X_features.iloc[split_idx:]
    y_train, y_test = y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:]

    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"  ✅ veg RandomForest → MAE={mae:.3f}, R²={r2:.3f} (train={len(X_train)}, test={len(X_test)})")

    model_dir = os.path.join(BASE_DIR, "final_models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "veg_global_random_forest.pkl")
    joblib.dump(rf, model_path)
    print(f"  💾 Saved FINAL veg model → {model_path}")

    metrics = {
        "kind": "veg",
        "item": None,
        "model": "Final_Global_RandomForest",
        "mae": mae,
        "r2": r2,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    return pd.DataFrame([metrics])


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------

def main():
    all_metrics = []

    # --- FRUITS: item-wise Linear Regression ---
    X_fruit, y_fruit, df_clean_fruit = load_dataset("fruit")
    fruit_metrics_df = train_fruit_itemwise_linear_regression(X_fruit, y_fruit)
    all_metrics.append(fruit_metrics_df)

    # --- VEGETABLES: global Random Forest ---
    X_veg, y_veg, df_clean_veg = load_dataset("veg")
    veg_metrics_df = train_veg_global_random_forest(X_veg, y_veg)
    all_metrics.append(veg_metrics_df)

    # Combine and save metrics
    final_metrics_df = pd.concat(all_metrics, ignore_index=True)
    metrics_path = os.path.join(BASE_DIR, "final_model_metrics.csv")
    final_metrics_df.to_csv(metrics_path, index=False)
    print(f"\n📊 Saved FINAL model metrics → {metrics_path}")
    print(final_metrics_df)


if __name__ == "__main__":
    main()

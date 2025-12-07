# # import pandas as pd
# # import json
# # import joblib
# # from sklearn.linear_model import LinearRegression

# # X_PATH = r"E:/vegteble_demand_forecasting/Data_preprocessing/X.csv"
# # Y_PATH = r"E:/vegteble_demand_forecasting/Data_preprocessing/y.csv"

# # Y_PATH = "Data_preprocessing/y.csv"
# # FEATURE_COLS_PATH = "Data_preprocessing/feature_columns.json"
# # ENCODER_PATH = "Data_preprocessing/encoder.pkl"
# # MODEL_PATH = "models/models.pkl"   # will store multiple models


# # def main():
# #     # Load full cleaned data (to know which row belongs to which item)
# #     df_full = pd.read_csv("sales_data_clean.csv")
# #     # Load processed features and target
# #     X = pd.read_csv(X_PATH)
# #     y = pd.read_csv(Y_PATH, header=None).iloc[:, 0]

# #     # Add the "item" column back (critical!)
# #     items = df_full["item"].reset_index(drop=True)
# #     X["item"] = items

# #     # Train one model per item
# #     unique_items = X["item"].unique()
# #     models = {}

# #     for item in unique_items:
# #         print(f"\n📌 Training model for: {item}")

# #         # Filter data belonging only to this item
# #         X_item = X[X["item"] == item].drop(columns=["item"])
# #         y_item = y.loc[X_item.index]

# #         # Train linear regression
# #         model = LinearRegression()
# #         model.fit(X_item, y_item)

# #         # Store model
# #         models[item] = model

# #         print(f"  ✔ Coefficients: {model.coef_[:5]} ...")
# #         print(f"  ✔ Intercept: {model.intercept_}")

# #     # Save the dictionary of models
# #     joblib.dump(models, MODEL_PATH)

# #     print("\n✅ Saved models for each item in 'models.pkl'")
# #     print("Available models:", list(models.keys()))


# # if __name__ == "__main__":
# #     main()

# import os
# import numpy as np
# import pandas as pd
# import joblib

# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, r2_score

# BASE_DIR = "../Model_Data"

# DATA_CONFIG = {
#     "fruit": {
#         "X_path": os.path.join(BASE_DIR, "fruit_X.csv"),
#         "y_path": os.path.join(BASE_DIR, "fruit_y.csv"),
#         "clean_path": "Data_preprocessing/fruit_sales_clean.csv",
#     },
#     "veg": {
#         "X_path": os.path.join(BASE_DIR, "veg_X.csv"),
#         "y_path": os.path.join(BASE_DIR, "veg_y.csv"),
#         "clean_path": "Data_preprocessing/vegetable_sales_clean.csv",
#     },
# }


# def load_dataset(kind: str):
#     cfg = DATA_CONFIG[kind]

#     X = pd.read_csv(cfg["X_path"])
#     y = pd.read_csv(cfg["y_path"], header=None).iloc[:, 0]
#     df_clean = pd.read_csv(cfg["clean_path"])
#     df_clean["date"] = pd.to_datetime(df_clean["date"])

#     assert len(X) == len(y) == len(df_clean), "Length mismatch between X, y, clean data"

#     # attach item + date so we can do time-based split and item-wise models
#     X["item"] = df_clean["item"].values
#     X["date"] = df_clean["date"].values

#     return X, y, df_clean


# # ------------------------------------------------------------------
# # 1) MOVING AVERAGE (item-wise, time-series style baseline)
# # ------------------------------------------------------------------

# def moving_average_itemwise(df_clean: pd.DataFrame, kind: str, window: int = 7, test_ratio: float = 0.2):
#     """
#     Simple time-series baseline:
#     For each item:
#       - sort by date
#       - split last `test_ratio` period as test
#       - predict each test-day demand as the mean of the previous `window` days
#     """
#     print(f"\n===== ⏳ Moving Average Baseline ({kind}) =====")

#     df = df_clean.copy()
#     df = df.sort_values(["item", "date"])

#     maes = []
#     details = []

#     for item in sorted(df["item"].unique()):
#         item_df = df[df["item"] == item].copy().sort_values("date")
#         n = len(item_df)
#         if n < window + 5:
#             print(f"  ⚠ Skipping {item}: not enough history ({n} rows)")
#             continue

#         split_idx = int((1 - test_ratio) * n)
#         train_df = item_df.iloc[:split_idx]
#         test_df = item_df.iloc[split_idx:]

#         # compute rolling mean on train+test but only use predictions for test part
#         item_df["ma_window"] = item_df["quantity_sold"].rolling(window=window, min_periods=1).mean()

#         # align predictions for test dates
#         ma_pred = item_df["ma_window"].iloc[split_idx:]
#         y_true = test_df["quantity_sold"]

#         mae = mean_absolute_error(y_true, ma_pred)
#         maes.append(mae)
#         details.append({"item": item, "mae": mae})

#         print(f"  {item}: MAE={mae:.3f} (n_test={len(y_true)})")

#     if maes:
#         print(f"\n👉 {kind} Moving Average (window={window}) – Mean MAE across items: {np.mean(maes):.3f}")
#     else:
#         print("No items evaluated for moving average.")

#     return pd.DataFrame(details)


# # ------------------------------------------------------------------
# # 2) GLOBAL MODELS (non item-wise): Linear Regression & Random Forest
# # ------------------------------------------------------------------

# def global_models(X: pd.DataFrame, y: pd.Series, kind: str):
#     """
#     Train + evaluate non-item-wise models:
#       - Linear Regression
#       - Random Forest
#     with time-based split on date.
#     """
#     print(f"\n===== 🌍 Global Models ({kind}) =====")

#     # Sort by date for time split
#     X_sorted = X.sort_values("date")
#     y_sorted = y.loc[X_sorted.index]

#     # drop helper cols from features
#     X_features = X_sorted.drop(columns=["item", "date"])

#     n = len(X_features)
#     split_idx = int(0.8 * n)

#     X_train, X_test = X_features.iloc[:split_idx], X_features.iloc[split_idx:]
#     y_train, y_test = y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:]

#     results = []

#     # 1) Linear Regression
#     linreg = LinearRegression()
#     linreg.fit(X_train, y_train)
#     y_pred_lr = linreg.predict(X_test)
#     mae_lr = mean_absolute_error(y_test, y_pred_lr)
#     r2_lr = r2_score(y_test, y_pred_lr)
#     print(f"  LinearRegression → MAE={mae_lr:.3f}, R²={r2_lr:.3f}")
#     results.append({"kind": kind, "model": "Global_LinearRegression", "mae": mae_lr, "r2": r2_lr})

#     # 2) Random Forest (extra ML approach)
#     rf = RandomForestRegressor(
#         n_estimators=200,
#         random_state=42,
#         n_jobs=-1
#     )
#     rf.fit(X_train, y_train)
#     y_pred_rf = rf.predict(X_test)
#     mae_rf = mean_absolute_error(y_test, y_pred_rf)
#     r2_rf = r2_score(y_test, y_pred_rf)
#     print(f"  RandomForest → MAE={mae_rf:.3f}, R²={r2_rf:.3f}")
#     results.append({"kind": kind, "model": "Global_RandomForest", "mae": mae_rf, "r2": r2_rf})

#     # you can save the best model if you want:
#     model_dir = os.path.join(BASE_DIR, "global_models")
#     os.makedirs(model_dir, exist_ok=True)
#     joblib.dump(linreg, os.path.join(model_dir, f"{kind}_global_linreg.pkl"))
#     joblib.dump(rf, os.path.join(model_dir, f"{kind}_global_rf.pkl"))
#     print(f"  ✅ Saved global models to {model_dir}")

#     return pd.DataFrame(results)


# # ------------------------------------------------------------------
# # 3) ITEM-WISE LINEAR REGRESSION (one model per item)
# # ------------------------------------------------------------------

# def itemwise_linear_regression(X: pd.DataFrame, y: pd.Series, kind: str):
#     """
#     Train a separate Linear Regression model for each item.
#     Time-based split within each item.
#     """
#     print(f"\n===== 🧺 Item-wise Linear Regression ({kind}) =====")

#     models = {}
#     metrics = []

#     for item in sorted(X["item"].unique()):
#         X_item = X[X["item"] == item].copy().sort_values("date")
#         y_item = y.loc[X_item.index]

#         X_features = X_item.drop(columns=["item", "date"])

#         n = len(X_features)
#         if n < 20:
#             print(f"  ⚠ Skipping {item}: not enough data ({n}).")
#             continue

#         split_idx = int(0.8 * n)
#         X_train, X_test = X_features.iloc[:split_idx], X_features.iloc[split_idx:]
#         y_train, y_test = y_item.iloc[:split_idx], y_item.iloc[split_idx:]

#         model = LinearRegression()
#         model.fit(X_train, y_train)

#         y_pred = model.predict(X_test)
#         mae = mean_absolute_error(y_test, y_pred)
#         r2 = r2_score(y_test, y_pred)

#         metrics.append({"kind": kind, "item": item, "model": "Item_LR", "mae": mae, "r2": r2})
#         models[item] = model

#         print(f"  {item} → MAE={mae:.3f}, R²={r2:.3f}")

#     # Save models
#     out_path = os.path.join(BASE_DIR, f"{kind}_item_linreg_models.pkl")
#     joblib.dump(models, out_path)
#     print(f"✅ Saved item-wise {kind} LR models → {out_path}")

#     return pd.DataFrame(metrics)


# # ------------------------------------------------------------------
# # MAIN: run all experiments for fruits & veg
# # ------------------------------------------------------------------

# def main():
#     all_results = []

#     for kind in ["fruit", "veg"]:
#         X, y, df_clean = load_dataset(kind)

#         # 1) Moving average baseline (item-wise, time-series)
#         ma_df = moving_average_itemwise(df_clean, kind, window=7)
#         ma_df["kind"] = kind
#         ma_df["model"] = "MovingAverage_itemwise"
#         all_results.append(ma_df)

#         # 2) Global models (non item-wise)
#         global_df = global_models(X, y, kind)
#         all_results.append(global_df)

#         # 3) Item-wise Linear Regression
#         item_lr_df = itemwise_linear_regression(X, y, kind)
#         all_results.append(item_lr_df)

#     # Concatenate & save all metrics
#     all_results_df = pd.concat(all_results, ignore_index=True, sort=False)
#     # out_metrics_path = os.path.join(BASE_DIR, "all_model_experiments_metrics.csv")
#     out_metrics_path = ("all_model_experiments_metrics.csv")
#     all_results_df.to_csv(out_metrics_path, index=False)
#     print(f"\n📊 Saved all experiment metrics → {out_metrics_path}")


# if __name__ == "__main__":
#     main()

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

# model/predict_next_7_days.py

import os
import json
from datetime import timedelta

import numpy as np
import pandas as pd
import joblib

# ---------- CONFIG ----------

BASE_DIR = "Model_Data"

DATA_CONFIG = {
    "fruit": {
        # Clean historical data (after preprocessing)
        "clean_path": "Data_preprocessing/fruit_sales_clean.csv",

        # Preprocessing artifacts
        "encoder_path": os.path.join(BASE_DIR, "fruit_encoder.pkl"),
        "scaler_path": os.path.join(BASE_DIR, "fruit_scaler.pkl"),
        "feature_cols_path": os.path.join(BASE_DIR, "fruit_feature_cols.json"),

        # FINAL MODEL: item-wise Linear Regression
        "final_model_type": "itemwise_lr",
        "final_model_path": os.path.join(
            BASE_DIR, "final_models", "fruit_itemwise_linear_regression.pkl"
        ),

        # Where to save predictions
        "pred_out": "7_days_prediction/fruit_7day_predictions.csv",
    },
    "veg": {
        # Clean historical data (after preprocessing)
        "clean_path": "Data_preprocessing/vegetable_sales_clean.csv",

        # Preprocessing artifacts
        "encoder_path": os.path.join(BASE_DIR, "veg_encoder.pkl"),
        "scaler_path": os.path.join(BASE_DIR, "veg_scaler.pkl"),
        "feature_cols_path": os.path.join(BASE_DIR, "veg_feature_cols.json"),

        # FINAL MODEL: one global Random Forest
        "final_model_type": "global_rf",
        "final_model_path": os.path.join(
            BASE_DIR, "final_models", "veg_global_random_forest.pkl"
        ),

        # Where to save predictions
        "pred_out": "7_days_prediction/veg_7day_predictions.csv",
    },
}


# ---------- HELPERS ----------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure we have standard column names:
      date, item, quantity_sold, price, weather, is_event, is_weekend, ...
    Works even if the CSV uses original raw names.
    """
    rename_map = {
        "Date": "date",
        "Item": "item",
        "Quantity_Sold": "quantity_sold",
        "Quantity_Sold(In Kg)": "quantity_sold",
        "Price": "price",
        "Price/Kg(INR)": "price",
        "Weather": "weather",
        "Is_Event": "is_event",
        "Is_Weekend": "is_weekend",
    }

    cols = {c: rename_map.get(c, c) for c in df.columns}
    df = df.rename(columns=cols)
    return df


def month_to_season(m: int) -> str:
    """Same season mapping used during preprocessing."""
    if m in [11, 12, 1, 2]:
        return "Winter"
    elif m in [3, 4, 5, 6]:
        return "Summer"
    elif m in [7, 8, 9]:
        return "Rainy"
    else:
        return "Transition"  # October


def build_future_raw_features(df_clean: pd.DataFrame, horizon_days: int = 7) -> pd.DataFrame:
    """
    Build a raw feature dataframe for the next `horizon_days` for each item.

    Assumptions:
      - Forecast for consecutive days after the last historical date.
      - For each item, use its last observed price.
      - Weather = most frequent historical weather.
      - No special events (is_event = 0).
    """
    df = normalize_columns(df_clean.copy())
    df["date"] = pd.to_datetime(df["date"])

    last_date = df["date"].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, horizon_days + 1)]

    items = df["item"].unique()

    # Last price per item (fallback to global median if missing)
    last_price_per_item = (
        df.sort_values("date")
          .groupby("item")["price"]
          .last()
    )

    mode_weather = df["weather"].mode()
    default_weather = mode_weather.iloc[0] if len(mode_weather) else "Sunny"

    rows = []
    for d in future_dates:
        for item in items:
            price = last_price_per_item.get(item, df["price"].median())
            day_of_week = d.weekday()
            month = d.month
            season = month_to_season(month)
            is_weekend = 1 if day_of_week >= 5 else 0

            rows.append(
                {
                    "date": d,
                    "item": item,
                    "price": price,
                    "is_event": 0,   # can be extended later
                    "is_weekend": is_weekend,
                    "day_of_week": day_of_week,
                    "month": month,
                    "weather": default_weather,
                    "season": season,
                }
            )

    return pd.DataFrame(rows)


def preprocess_future_features(future_raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Apply the SAME preprocessing as during training:

      - OneHotEncoder on (item, weather, season)
      - StandardScaler on all features
      - Ensure column order matches training (feature_cols.json)
    """
    encoder = joblib.load(cfg["encoder_path"])
    scaler = joblib.load(cfg["scaler_path"])

    with open(cfg["feature_cols_path"], "r") as f:
        feature_cols = json.load(f)

    categorical_cols = ["item", "weather", "season"]
    numeric_cols = ["price", "is_event", "is_weekend", "day_of_week", "month"]

    # Encode categoricals using the fitted encoder
    encoded = encoder.transform(future_raw[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=future_raw.index)

    # Numeric data
    numeric_df = future_raw[numeric_cols]

    # Combine numeric + encoded
    X_raw = pd.concat([numeric_df, encoded_df], axis=1)

    # Ensure all training feature columns exist
    missing_cols = [c for c in feature_cols if c not in X_raw.columns]
    for col in missing_cols:
        X_raw[col] = 0.0  # category not present in future

    # Reorder columns to match training
    X_raw = X_raw[feature_cols]

    # Scale
    X_scaled = scaler.transform(X_raw)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=future_raw.index)

    return X_scaled


def predict_for_kind(kind: str, cfg: dict, horizon_days: int = 7):
    print(f"\n==============================")
    print(f"🔮 Predicting next {horizon_days} days for: {kind.upper()}")
    print(f"==============================")

    # 1. Load clean historical data
    df_clean = pd.read_csv(cfg["clean_path"])
    df_clean = normalize_columns(df_clean)
    df_clean["date"] = pd.to_datetime(df_clean["date"])

    # 2. Build future raw feature rows (date, item, price, weather, etc.)
    future_raw = build_future_raw_features(df_clean, horizon_days=horizon_days)

    # 3. Preprocess to same X-format as training
    X_future = preprocess_future_features(future_raw, cfg)

    # 4. Baseline: Moving Average per item (last 7 days)
    window = 7
    baseline_preds = []
    for _, row in future_raw.iterrows():
        item = row["item"]
        hist_item = (
            df_clean[df_clean["item"] == item]
            .sort_values("date")
            .tail(window)["quantity_sold"]
        )
        if len(hist_item) == 0:
            ma_val = df_clean["quantity_sold"].tail(window).mean()
        else:
            ma_val = hist_item.mean()

        baseline_preds.append(ma_val)

    future_raw["baseline_moving_avg"] = baseline_preds

    # 5. Load final model and predict
    final_model_type = cfg["final_model_type"]
    final_model_path = cfg["final_model_path"]

    print(f"  Using final model type: {final_model_type}")
    print(f"  Loading model from: {final_model_path}")

    if final_model_type == "itemwise_lr":
        # This is a dict: item -> LinearRegression
        item_models = joblib.load(final_model_path)
        preds = []

        for idx, row in future_raw.iterrows():
            item = row["item"]
            model = item_models.get(item)

            if model is None:
                # Fallback: use baseline if somehow model is missing
                preds.append(future_raw.loc[idx, "baseline_moving_avg"])
            else:
                x_row = X_future.loc[[idx]]  # single row DF
                pred_val = model.predict(x_row)[0]
                preds.append(pred_val)

        future_raw["predicted_quantity(in Kg)"] = preds

    elif final_model_type == "global_rf":
        rf_model = joblib.load(final_model_path)
        preds = rf_model.predict(X_future)
        future_raw["predicted_quantity(in Kg)"] = preds

    else:
        raise ValueError(f"Unknown final_model_type: {final_model_type}")

    # 6. Clip + round (no negative demand)
    for col in ["baseline_moving_avg", "predicted_quantity(in Kg)"]:
        future_raw[col] = np.clip(future_raw[col], a_min=0, a_max=None)
        future_raw[col] = future_raw[col].round(0).astype(int)

    # 7. Save predictions
    out_path = cfg["pred_out"]
    future_raw.to_csv(out_path, index=False)
    print(f"✅ Saved {kind} 7-day predictions → {out_path}")
    print("\nSample:")
    print(future_raw.head())


def main():
    for kind, cfg in DATA_CONFIG.items():
        predict_for_kind(kind, cfg, horizon_days=7)


if __name__ == "__main__":
    main()


# demand_api.py

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify

# ----------------- CONFIG -----------------

BASE_DIR = "Model_Data"  # Directory where models and encoders live

DATA_CONFIG = {
    "fruit": {
        "encoder_path": os.path.join(BASE_DIR, "fruit_encoder.pkl"),
        "scaler_path": os.path.join(BASE_DIR, "fruit_scaler.pkl"),
        "feature_cols_path": os.path.join(BASE_DIR, "fruit_feature_cols.json"),
        "item_models_path": os.path.join(BASE_DIR, "final_models", "fruit_itemwise_linear_regression.pkl"),
        "existing_pred_path": "7_days_prediction/fruit_7day_predictions.csv",
    },
    "veg": {
        "encoder_path": os.path.join(BASE_DIR, "veg_encoder.pkl"),
        "scaler_path": os.path.join(BASE_DIR, "veg_scaler.pkl"),
        "feature_cols_path": os.path.join(BASE_DIR, "veg_feature_cols.json"),
        "rf_model_path": os.path.join(BASE_DIR, "final_models", "veg_global_random_forest.pkl"),
        "existing_pred_path": "7_days_prediction/veg_7day_predictions.csv",
    },
}

app = Flask(__name__)


# ----------------- HELPER FUNCTIONS -----------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names so the logic works with raw or clean CSVs."""
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
    return df.rename(columns=cols)


def month_to_season(m: int) -> str:
    if m in [11, 12, 1, 2]:
        return "Winter"
    elif m in [3, 4, 5, 6]:
        return "Summer"
    elif m in [7, 8, 9]:
        return "Rainy"
    else:
        return "Transition"  # October


def ensure_calendar_and_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist and are consistent with training preprocessing."""
    if "date" not in df.columns:
        raise ValueError("Input must contain a 'date' or 'Date' column.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df["day_of_week"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].apply(month_to_season)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Event flag
    if "is_event" not in df.columns:
        df["is_event"] = 0
    else:
        df["is_event"] = df["is_event"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    # Weather
    if "weather" not in df.columns:
        df["weather"] = "Sunny"
    df["weather"] = df["weather"].fillna("Sunny")

    # Price
    if "price" not in df.columns:
        raise ValueError("Input must contain 'price' or 'Price/Kg(INR)' column.")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["price"] = df["price"].fillna(df["price"].median())

    # Item
    if "item" not in df.columns:
        raise ValueError("Input must contain 'item' or 'Item' column.")

    return df


def build_feature_matrix(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Apply training-time OneHotEncoder + StandardScaler to new data."""
    cfg = DATA_CONFIG[kind]

    encoder = joblib.load(cfg["encoder_path"])
    scaler = joblib.load(cfg["scaler_path"])

    with open(cfg["feature_cols_path"], "r") as f:
        feature_cols = json.load(f)

    categorical_cols = ["item", "weather", "season"]
    numeric_cols = ["price", "is_event", "is_weekend", "day_of_week", "month"]

    encoded = encoder.transform(df[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)

    numeric_df = df[numeric_cols]
    X_raw = pd.concat([numeric_df, encoded_df], axis=1)

    # Add any missing columns as 0
    missing_cols = [c for c in feature_cols if c not in X_raw.columns]
    for col in missing_cols:
        X_raw[col] = 0.0

    X_raw = X_raw[feature_cols]

    X_scaled = scaler.transform(X_raw)
    return pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)


def predict_with_best_model(kind: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply BEST model:
      - fruit → item-wise Linear Regression
      - veg   → global RandomForest
    Returns df with 'predicted_quantity_sold' column.
    """
    df = normalize_columns(df)
    df = ensure_calendar_and_flags(df)
    X_new = build_feature_matrix(df, kind=kind)

    if kind == "fruit":
        # Item-wise Linear Regression
        cfg = DATA_CONFIG["fruit"]
        item_models = joblib.load(cfg["item_models_path"])
        preds = []

        for idx, row in df.iterrows():
            item = row["item"]
            model = item_models.get(item)
            if model is None:
                preds.append(np.nan)
            else:
                x_row = X_new.loc[[idx]]
                preds.append(model.predict(x_row)[0])

    else:
        # Global RandomForest
        cfg = DATA_CONFIG["veg"]
        rf_model = joblib.load(cfg["rf_model_path"])
        preds = rf_model.predict(X_new)

    df["predicted_quantity_sold"] = (
        np.clip(preds, a_min=0, a_max=None)
        .round(0)
        .astype("Int64")
    )
    return df


# ----------------- API ROUTES -----------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": datetime.now().isoformat()})


@app.route("/predictions/7day/<kind>", methods=["GET"])
def get_existing_predictions(kind):
    if kind not in DATA_CONFIG:
        return jsonify({"error": "kind must be 'fruit' or 'veg'"}), 400

    path = DATA_CONFIG[kind]["existing_pred_path"]
    if not os.path.exists(path):
        return jsonify({"error": f"No existing prediction file found at {path}"}), 404

    df = pd.read_csv(path)
    return jsonify(df.to_dict(orient="records"))


@app.route("/predict/<kind>", methods=["POST"])
def predict_new():
    if kind := request.view_args.get("kind") not in DATA_CONFIG:
        return jsonify({"error": "kind must be 'fruit' or 'veg'"}), 400

    kind = request.view_args["kind"]

    if "file" not in request.files:
        return jsonify({"error": "Upload a CSV file in form field 'file'"}), 400

    file = request.files["file"]
    try:
        df_input = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV: {str(e)}"}), 400

    try:
        df_pred = predict_with_best_model(kind, df_input)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify(df_pred.to_dict(orient="records"))


if __name__ == "__main__":
    # Run: python demand_api.py
    app.run(host="0.0.0.0", port=8000, debug=True)

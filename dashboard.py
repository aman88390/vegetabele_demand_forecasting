# dashboard.py

import os
import json

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ----------------- CONFIG -----------------

BASE_DIR = "Model_Data"

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


# ----------------- SHARED HELPERS -----------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
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
        return "Transition"


def ensure_calendar_and_flags(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        raise ValueError("Input must contain 'date' or 'Date' column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df["day_of_week"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].apply(month_to_season)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    if "is_event" not in df.columns:
        df["is_event"] = 0
    else:
        df["is_event"] = df["is_event"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    if "weather" not in df.columns:
        df["weather"] = "Sunny"
    df["weather"] = df["weather"].fillna("Sunny")

    if "price" not in df.columns:
        raise ValueError("Input must contain 'price' or 'Price/Kg(INR)' column.")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["price"] = df["price"].fillna(df["price"].median())

    if "item" not in df.columns:
        raise ValueError("Input must contain 'item' or 'Item' column.")

    return df


def build_feature_matrix(df: pd.DataFrame, kind: str) -> pd.DataFrame:
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

    missing_cols = [c for c in feature_cols if c not in X_raw.columns]
    for col in missing_cols:
        X_raw[col] = 0.0

    X_raw = X_raw[feature_cols]

    X_scaled = scaler.transform(X_raw)
    return pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)


def predict_with_best_model(kind: str, df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    df = ensure_calendar_and_flags(df)
    X_new = build_feature_matrix(df, kind=kind)

    if kind == "fruit":
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
        cfg = DATA_CONFIG["veg"]
        rf_model = joblib.load(cfg["rf_model_path"])
        preds = rf_model.predict(X_new)

    df["predicted_quantity_sold(in Kg)"] = (
        np.clip(preds, a_min=0, a_max=None)
        .round(0)
        .astype(int)
    )
    return df


# ----------------- STREAMLIT APP -----------------

st.set_page_config(page_title="Street Vendor Demand Dashboard", layout="wide")
st.title("🥕 Street Vendor Demand Prediction Dashboard")

tab1, tab2 = st.tabs(["📈 Existing 7-day Forecast", "📤 Predict from CSV"])


# -------- TAB 1: EXISTING 7-DAY FORECAST --------

with tab1:
    st.subheader("Existing 7-day Demand Forecast")

    kind = st.selectbox("Select dataset", ["fruit", "veg"])

    cfg = DATA_CONFIG[kind]
    pred_path = cfg["existing_pred_path"]

    if os.path.exists(pred_path):
        df_pred = pd.read_csv(pred_path)
        df_pred = normalize_columns(df_pred)

        # Try to detect date / prediction columns
        if "date" in df_pred.columns:
            df_pred["date"] = pd.to_datetime(df_pred["date"], errors="coerce")

        st.write(f"Showing file: `{pred_path}`")
        st.dataframe(df_pred.head())

        # ---- Decide which prediction column to use (best model per kind) ----
        # Fruit → prefer item-wise LR col; Veg → prefer global RF col
        if kind == "fruit":
            # common names we may have
            candidate_cols = [
                "pred_itemwise_linreg",
                "predicted_quantity_sold(in Kg)",
                "predicted_quantity_sold",
            ]
        else:
            candidate_cols = [
                "pred_global_rf",
                "predicted_quantity_sold(in Kg)",
                "predicted_quantity_sold",
            ]

        # Auto-pick first that exists, else fall back to "any col starting with 'pred'"
        pred_col = None
        for c in candidate_cols:
            if c in df_pred.columns:
                pred_col = c
                break

        if pred_col is None:
            # fallback: first column whose name starts with 'pred'
            pred_candidates = [c for c in df_pred.columns if c.lower().startswith("pred")]
            if pred_candidates:
                pred_col = pred_candidates[0]

        if pred_col is None:
            st.warning("Could not detect a prediction column in existing CSV.")
        else:
            st.success(f"Using prediction column: **{pred_col}**")

            # ---- Overall line chart (sum over all items) ----
            if "date" in df_pred.columns:
                st.markdown("### 📊 Overall Forecast (All Items)")
                overall_series = (
                    df_pred.groupby("date")[pred_col]
                    .sum()
                    .sort_index()
                )
                st.line_chart(overall_series)

            # ---- Item-wise forecast ----
            if "item" in df_pred.columns:
                st.markdown("### 🧺 Item-wise Forecast")

                items = sorted(df_pred["item"].unique().tolist())
                selected_item = st.selectbox("Select item", items)

                item_df = df_pred[df_pred["item"] == selected_item].copy()
                if "date" in item_df.columns:
                    item_series = (
                        item_df.groupby("date")[pred_col]
                        .sum()
                        .sort_index()
                    )
                    st.line_chart(item_series)

                # Top items across 7-day horizon
                st.markdown("#### Top items over 7-day forecast horizon")
                top_items = (
                    df_pred.groupby("item")[pred_col]
                    .sum()
                    .sort_values(ascending=False)
                    .head(10)
                )
                st.bar_chart(top_items)
            else:
                st.info("No 'item' column found in existing predictions, cannot show item-wise plots.")
    else:
        st.warning(
            f"No existing prediction file found at `{pred_path}`. "
            f"Run your 7-day prediction script first."
        )


# -------- TAB 2: UPLOAD CSV & PREDICT --------

with tab2:
    st.subheader("Upload a CSV to Predict Demand")

    kind2 = st.selectbox("Kind of data", ["fruit", "veg"], key="kind2")
    uploaded_file = st.file_uploader(
        "Upload a CSV file with at least: date, item, price (weather/is_event optional)",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df_input.head())

            df_pred = predict_with_best_model(kind2, df_input)

            st.success("Predictions generated!")
            st.dataframe(df_pred.head())

            # Visualizations
            if "date" in df_pred.columns:
                df_pred["date"] = pd.to_datetime(df_pred["date"], errors="coerce")

            if "predicted_quantity_sold(in Kg)" in df_pred.columns and "date" in df_pred.columns:
                st.markdown("### 📊 Overall Predicted Demand (All Items)")
                overall_uploaded = (
                    df_pred.groupby("date")["predicted_quantity_sold(in Kg)"]
                    .sum()
                    .sort_index()
                )
                st.line_chart(overall_uploaded)

            if "item" in df_pred.columns and "predicted_quantity_sold(in Kg)" in df_pred.columns:
                st.markdown("### 🧺 Item-wise Predicted Demand (Last Date)")
                latest_date = df_pred["date"].max()
                latest_df = df_pred[df_pred["date"] == latest_date]
                by_item_latest = (
                    latest_df.groupby("item")["predicted_quantity_sold(in Kg)"]
                    .sum()
                    .sort_values(ascending=False)
                )
                st.caption(f"Predicted demand on last date in file: {latest_date.date()}")
                st.bar_chart(by_item_latest)

            # Download predictions
            csv_bytes = df_pred.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download predictions as CSV",
                data=csv_bytes,
                file_name=f"{kind2}_predictions_from_upload.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Error while predicting: {e}")

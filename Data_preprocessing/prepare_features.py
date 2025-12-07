# import pandas as pd
# import json
# import joblib
# from sklearn.preprocessing import OneHotEncoder, StandardScaler

# CLEAN_PATH = "sales_data_clean.csv"
# X_OUT = "X.csv"
# Y_OUT = "y.csv"
# SCALER_OUT = "scaler.pkl"
# ENCODER_OUT = "encoder.pkl"
# FEATURE_COLS_OUT = "feature_columns.json"


# def main():
#     # Load cleaned data
#     df = pd.read_csv(CLEAN_PATH)
#     df["date"] = pd.to_datetime(df["date"])

#     # --------------------------
#     # 1. Select useful features
#     # --------------------------
#     feature_df = df[
#         [
#             "item",
#             "price",
#             "is_event",
#             "is_weekend",
#             "day_of_week",
#             "month",
#             "weather",
#             "season",
#         ]
#     ]
#     target = df["quantity_sold"]

#     # ----------------------------------
#     # 2. One-hot encode categorical vars
#     # ----------------------------------
#     categorical_cols = ["item", "weather", "season"]
#     encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

#     encoded_array = encoder.fit_transform(feature_df[categorical_cols])
#     encoded_cols = encoder.get_feature_names_out(categorical_cols)

#     encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols)

#     # ------------------------------------
#     # 3. Combine with numeric features
#     # ------------------------------------
#     numeric_cols = ["price", "is_event", "is_weekend", "day_of_week", "month"]
#     numeric_df = feature_df[numeric_cols].reset_index(drop=True)

#     # Combined DataFrame (before scaling)
#     X = pd.concat([numeric_df, encoded_df], axis=1)

#     # ------------------------------------
#     # 4. Scale all features
#     # ------------------------------------
#     scaler = StandardScaler()
#     X_scaled_array = scaler.fit_transform(X)

#     # Convert scaled numpy array back to pandas DataFrame
#     X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns)

#     # ------------------------------------
#     # 5. Save outputs
#     # ------------------------------------
#     X_scaled.to_csv(X_OUT, index=False)
#     target.to_csv(Y_OUT, index=False)

#     joblib.dump(scaler, SCALER_OUT)
#     joblib.dump(encoder, ENCODER_OUT)

#     # Save feature names
#     json.dump(list(X.columns), open(FEATURE_COLS_OUT, "w"), indent=2)

#     print("✅ Feature preprocessing complete.")
#     print("Saved:")
#     print(" - X.csv (scaled features)")
#     print(" - y.csv (target)")
#     print(" - scaler.pkl")
#     print(" - encoder.pkl")
#     print(" - feature_columns.json")
#     print("\nFeature sample:")
#     print(X_scaled.head())


# if __name__ == "__main__":
#     main()


import pandas as pd
import numpy as np
import json
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os

# Paths
FRUIT_CLEAN = "Data_preprocessing/fruit_sales_clean.csv"
VEG_CLEAN = "Data_preprocessing/vegetable_sales_clean.csv"

OUTPUT_DIR = "Model_Data"   # folder for prepared ML data
os.makedirs(OUTPUT_DIR, exist_ok=True)

def prepare_dataset(clean_path, prefix):
    print(f"\n🍏 Preparing dataset: {clean_path}")

    df = pd.read_csv(clean_path)
    df["date"] = pd.to_datetime(df["date"])

    # --------------------------
    # Select Features & Target
    # ---------------------------
    feature_df = df[
        [
            "item",
            "price",
            "is_event",
            "is_weekend",
            "day_of_week",
            "month",
            "weather",
            "season",
        ]
    ]

    target = df["quantity_sold"]

    # --------------------------
    # Categorical Encoding
    # --------------------------
    categorical_cols = ["item", "weather", "season"]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    encoded = encoder.fit_transform(feature_df[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)

    # --------------------------
    # Numeric Features
    # --------------------------
    numeric_cols = ["price", "is_event", "is_weekend", "day_of_week", "month"]
    numeric_data = feature_df[numeric_cols].values

    # --------------------------
    # Combine (before scaling)
    # --------------------------
    X = pd.DataFrame(
        data = np.hstack([numeric_data, encoded]),
        columns = list(numeric_cols) + list(encoded_cols)
    )

    # --------------------------
    # Scaling
    # --------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # --------------------------
    # Save outputs
    # --------------------------
    X_path = f"{OUTPUT_DIR}/{prefix}_X.csv"
    y_path = f"{OUTPUT_DIR}/{prefix}_y.csv"
    encoder_path = f"{OUTPUT_DIR}/{prefix}_encoder.pkl"
    scaler_path = f"{OUTPUT_DIR}/{prefix}_scaler.pkl"
    feature_cols_path = f"{OUTPUT_DIR}/{prefix}_feature_cols.json"

    X_scaled.to_csv(X_path, index=False)
    target.to_csv(y_path, index=False, header=False)
    joblib.dump(encoder, encoder_path)
    joblib.dump(scaler, scaler_path)

    json.dump(list(X.columns), open(feature_cols_path, "w"), indent=2)

    print(f"✅ Features saved:")
    print(f" - {X_path}")
    print(f" - {y_path}")
    print(f" - {encoder_path}")
    print(f" - {scaler_path}")
    print(f" - {feature_cols_path}")


def main():
    # Prepare fruits
    prepare_dataset(FRUIT_CLEAN, "fruit")

    # Prepare vegetables
    prepare_dataset(VEG_CLEAN, "veg")


if __name__ == "__main__":
    main()

# import pandas as pd
# import numpy as np

# RAW_PATH = "sales_data_raw.csv"
# CLEAN_PATH = "sales_data_clean.csv"


# def month_to_season(m):
#     """Simple season mapping based on Indian market patterns."""
#     if m in [11, 12, 1, 2]:
#         return "Winter"      # onion/potato cheap & abundant
#     elif m in [3, 4, 5, 6]:
#         return "Summer"      # tomatoes abundant
#     elif m in [7, 8, 9]:
#         return "Rainy"
#     else:
#         return "Transition"  # October


# def main():
#     # 1. Load raw data
#     df = pd.read_csv(RAW_PATH)
#     print("Raw shape:", df.shape)

#     # 2. Parse dates (drop rows with invalid/missing dates)
#     df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
#     df = df.dropna(subset=["Date"])

#     # 3. Standardise column names
#     df = df.rename(
#         columns={
#             "Date": "date",
#             "Item": "item",
#             "Quantity_Sold": "quantity_sold",
#             "Price": "price",
#             "Is_Event": "is_event_raw",
#             "Is_Weekend": "is_weekend_raw",
#             "Weather": "weather_raw",
#         }
#     )

#     # 4. Remove exact duplicate rows
#     before = len(df)
#     df = df.drop_duplicates()
#     print("Removed duplicates:", before - len(df))

#     # 5. Handle item – drop rows with missing item
#     missing_items = df["item"].isna().sum()
#     if missing_items > 0:
#         print("Dropping rows with missing item:", missing_items)
#         df = df.dropna(subset=["item"])

#     # 6. Clean event flag (Yes/No/Maybe -> 1/0)
#     df["is_event"] = df["is_event_raw"].map({"Yes": 1, "No": 0})
#     # Treat "Maybe" or anything else as non-event (0)
#     df["is_event"] = df["is_event"].fillna(0).astype(int)

#     # 7. Recompute weekend from date (more reliable than raw text)
#     df["day_of_week"] = df["date"].dt.weekday  # 0=Mon,...,6=Sun
#     df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

#     # 8. Clean weather
#     valid_weather = ["Sunny", "Cloudy", "Rain"]
#     df["weather"] = df["weather_raw"].where(df["weather_raw"].isin(valid_weather), np.nan)
#     # Fill invalid/missing with mode of valid values
#     mode_weather = df["weather"].mode()
#     fallback_weather = mode_weather.iloc[0] if len(mode_weather) else "Sunny"
#     df["weather"] = df["weather"].fillna(fallback_weather)

#     # 9. Clean quantity_sold
#     df["quantity_sold"] = pd.to_numeric(df["quantity_sold"], errors="coerce")
#     # Mark negative or absurdly large as NaN
#     df.loc[df["quantity_sold"] < 0, "quantity_sold"] = np.nan
#     df.loc[df["quantity_sold"] > 2000, "quantity_sold"] = np.nan  # treat 9999 as outlier

#     # Impute quantity per item with median
#     def impute_quantity(group):
#         med = group["quantity_sold"].median()
#         group["quantity_sold"] = group["quantity_sold"].fillna(med)
#         return group

#     df = df.groupby("item", group_keys=False).apply(impute_quantity)
#     df["quantity_sold"] = df["quantity_sold"].round().astype(int)

#     # 10. Clean price
#     df["price"] = pd.to_numeric(df["price"], errors="coerce")

#     # Impute missing price per item with median
#     def impute_price(group):
#         med = group["price"].median()
#         group["price"] = group["price"].fillna(med)
#         return group

#     df = df.groupby("item", group_keys=False).apply(impute_price)

#     # 11. Calendar features
#     df["month"] = df["date"].dt.month
#     df["season"] = df["month"].apply(month_to_season)

#     # 12. Sort by date and item (historical order)
#     df = df.sort_values(["date", "item"]).reset_index(drop=True)

#     # 13. Save clean data
#     df.to_csv(CLEAN_PATH, index=False)
#     print("✅ Saved cleaned data to:", CLEAN_PATH)
#     print("Clean shape:", df.shape)
#     print(df.head())


# if __name__ == "__main__":
#     main()

import pandas as pd
import numpy as np

def month_to_season(m: int) -> str:
    """Simple season mapping (can be mentioned in README)."""
    if m in [11, 12, 1, 2]:
        return "Winter"
    elif m in [3, 4, 5, 6]:
        return "Summer"
    elif m in [7, 8, 9]:
        return "Rainy"
    else:
        return "Transition"  # October


def preprocess_raw(input_path: str, output_path: str):
    print(f"\n📂 Loading: {input_path}")
    df = pd.read_csv(input_path)
    print("Raw shape:", df.shape)

    # 1. Parse Date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # 2. Standardise column names
    df = df.rename(
        columns={
            "Date": "date",
            "Item": "item",
            "Quantity_Sold(In Kg)": "quantity_sold",
            "Price/Kg(INR)": "price",
            "Is_Event": "is_event_raw",
            "Is_Weekend": "is_weekend_raw",
            "Weather": "weather_raw",
        }
    )

    # 3. Drop exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    print("Removed duplicates:", before - len(df))

    # 4. Drop rows without item
    missing_items = df["item"].isna().sum()
    if missing_items > 0:
        print("Dropping rows with missing item:", missing_items)
        df = df.dropna(subset=["item"])

    # 5. Clean event flag (Yes/No/Maybe -> 1/0, treat Maybe as 0)
    df["is_event"] = df["is_event_raw"].map({"Yes": 1, "No": 0})
    df["is_event"] = df["is_event"].fillna(0).astype(int)

    # 6. Recompute weekend from date (ignore raw text)
    df["day_of_week"] = df["date"].dt.weekday  # 0=Mon,...,6=Sun
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # 7. Clean weather (keep only Sunny/Cloudy/Rain)
    valid_weather = ["Sunny", "Cloudy", "Rain"]
    df["weather"] = df["weather_raw"].where(df["weather_raw"].isin(valid_weather), np.nan)
    mode_weather = df["weather"].mode()
    fallback_weather = mode_weather.iloc[0] if len(mode_weather) else "Sunny"
    df["weather"] = df["weather"].fillna(fallback_weather)

    # 8. Clean quantity_sold
    df["quantity_sold"] = pd.to_numeric(df["quantity_sold"], errors="coerce")
    # Negative or absurd values → NaN (base is ~15–40, events can push higher but not 999)
    df.loc[df["quantity_sold"] < 0, "quantity_sold"] = np.nan
    df.loc[df["quantity_sold"] > 200, "quantity_sold"] = np.nan

    def impute_quantity(group):
        med = group["quantity_sold"].median()
        group["quantity_sold"] = group["quantity_sold"].fillna(med)
        return group

    df = df.groupby("item", group_keys=False).apply(impute_quantity)
    df["quantity_sold"] = df["quantity_sold"].round().astype(int)

    # 9. Clean price
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df.loc[df["price"] <= 0, "price"] = np.nan

    def impute_price(group):
        med = group["price"].median()
        group["price"] = group["price"].fillna(med)
        return group

    df = df.groupby("item", group_keys=False).apply(impute_price)
    df["price"] = df["price"].round().astype(int)

    # 10. Calendar features
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].apply(month_to_season)

    # 11. Sort by date & item
    df = df.sort_values(["date", "item"]).reset_index(drop=True)

    # 12. Save cleaned data
    df.to_csv(output_path, index=False)
    print("✅ Saved cleaned data to:", output_path)
    print("Clean shape:", df.shape)
    print(df.head())

import os
print(os.listdir())


if __name__ == "__main__":
    # preprocess fruits
    preprocess_raw("Raw_Data/fruit_sales_raw.csv", "Data_preprocessing/fruit_sales_clean.csv")
    # preprocess vegetables
    preprocess_raw("Raw_Data/vegetable_sales_raw.csv", "Data_preprocessing/vegetable_sales_clean.csv")
# Raw_Data/generate_data.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# --- REPRODUCIBILITY ---
np.random.seed(42)
random.seed(42)

# --- CONFIGURATION ---
DAYS = 120  # ~4 months of data
END_DATE = datetime(2025, 11, 30)  # latest historical date
START_DATE = END_DATE - timedelta(days=DAYS - 1)
DATES = [START_DATE + timedelta(days=i) for i in range(DAYS)]

# 1. Common logic to define random high-demand event days
event_dates = set()
for _ in range(15):
    rand_day = np.random.randint(0, DAYS)
    event_dates.add((START_DATE + timedelta(days=rand_day)).strftime('%Y-%m-%d'))


def generate_dataset(items, veg_in_season, output_path):
    """
    items: dict of item_name -> {base_qty, base_price, type}
    veg_in_season: dict for seasonal veg items (only used for vegetables)
    output_path: CSV filename to save
    """
    data = []

    print(f"\nGenerating RAW data for: {output_path}")

    for date in DATES:
        date_str = date.strftime('%Y-%m-%d')
        month = date.month

        is_weekend = date.weekday() >= 5
        is_event = date_str in event_dates

        # Weather: same logic as before
        weather = np.random.choice(['Sunny', 'Cloudy', 'Rain'], p=[0.70, 0.20, 0.10])

        for item, profile in items.items():
            qty = profile['base_qty']
            price = profile['base_price']

            # --- SEASONALITY FOR VEGETABLES ONLY (internal) ---
            # Fruits won't be in veg_in_season, so they skip this block
            if profile['type'] == 'Vegetable':
                if item in veg_in_season and month in veg_in_season[item]:
                    # In-season: cheaper & more volume
                    qty *= 1.3
                    price *= 0.8
                else:
                    # Off-season: costly & lower supply
                    qty *= 0.8
                    price *= 1.15

            # --- WEEKEND EFFECT ---
            if is_weekend:
                # Fruits get a slightly higher weekend bump than vegetables
                if profile['type'] == 'Fruit':
                    qty *= 1.10
                else:
                    qty *= 1.10

            # --- EVENT LOGIC (High price AND high demand) ---
            if is_event:
                # Vendors increase price because demand is guaranteed
                price *= np.random.uniform(1.2, 1.3)
                # Demand jumps strongly despite high price
                event_multiplier = 2.0 if profile['type'] == 'Fruit' else 1.5
                qty *= event_multiplier

                # Weather on event days: rain hurts but doesn't kill demand
                if weather == 'Rain':
                    qty *= 0.90 # slight drop but still high
                elif weather == 'Sunny':
                    qty *= 1.05   # small extra boost

            else:
                # --- NORMAL MARKET PRICE-DEMAND RELATION ---
                # small random fluctuation in price
                price_shock = np.random.uniform(0.95,1.05)
                price *= price_shock

                # Price elasticity: if price low → demand high, price high → demand low
                price_ratio = price / profile['base_price']
                if price_ratio > 1.10:
                    qty *= 0.85   # more expensive than usual → fewer buyers
                elif price_ratio < 0.90:
                    qty *= 1.15   # cheaper than usual → more buyers

                # Weather effect strong on normal days
                if weather == 'Rain':
                    qty *= 0.40   # street sales badly hit
                elif weather == 'Sunny':
                    qty *= 1.05

            # --- RANDOM OUTLIERS (real-world chaos) ---
            rand_val = random.random()
            if rand_val < 0.03:
                # Bulk restaurant / catering order
                qty *= 2.0
            elif rand_val > 0.98:
                # Supply issue, late truck, etc.
                qty *= 0.1

            # --- FINAL NOISE & CLAMPING ---
            noise = np.random.normal(0, 3)  # small noise
            final_qty = int(max(0, qty + noise))
            final_price = int(price)

            row = [
                date_str,
                item,
                final_qty,
                final_price,
                "Yes" if is_event else "No",
                "Yes" if is_weekend else "No",
                weather
            ]
            data.append(row)

    # Build DataFrame
    columns = ["Date", "Item", "Quantity_Sold(In Kg)", "Price/Kg(INR)", "Is_Event", "Is_Weekend", "Weather"]
    df = pd.DataFrame(data, columns=columns)

    # --- RAW DATA CORRUPTION TO MAKE IT REALISTIC ---
    print("Injecting missing values, wrong values, and duplicates...")

    # 1. Random missing values (~3%)
    for col in ["Quantity_Sold(In Kg)", "Price/Kg(INR)", "Weather", "Item"]:
        mask = np.random.rand(len(df)) < 0.03
        df.loc[mask, col] = np.nan

    # 2. Wrong numeric values (~1%)
    wrong_idx = df.sample(frac=0.01, random_state=42).index
    df.loc[wrong_idx, "Quantity_Sold(In Kg)"] = np.random.choice([-3, -5, 999])  # still some bad values

    # 3. Wrong category values (~1%)
    df.loc[df.sample(frac=0.01, random_state=7).index, "Weather"] = "Storm"   # unexpected
    df.loc[df.sample(frac=0.01, random_state=123).index, "Is_Event"] = "Maybe"  # invalid

    # 4. Duplicate rows (~2%)
    duplicate_rows = df.sample(frac=0.02, random_state=99)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    # 5. Sort by Date (serial historical data)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ RAW dataset ready: '{output_path}'")


def main():
    # --- FRUIT VENDOR DATASET (5 fruit categories) ---
    fruit_items = {
        # base_qty in range ~10–35 (realistic per-day street vendor sales, in kg/units)
        "Apple":   {"base_qty": 20, "base_price": 140, "type": "Fruit"},
        "Banana":  {"base_qty": 30, "base_price": 40,  "type": "Fruit"},
        "Orange":  {"base_qty": 25, "base_price": 60,  "type": "Fruit"},
        "Grapes":  {"base_qty": 18, "base_price": 80,  "type": "Fruit"},
        "Mango":   {"base_qty": 15, "base_price": 100, "type": "Fruit"},
    }

    # No seasonality dict needed for fruits in this version
    fruit_veg_in_season = {}

    generate_dataset(
        items=fruit_items,
        veg_in_season=fruit_veg_in_season,
        output_path="Raw_Data/fruit_sales_raw.csv",
    )

    # --- VEGETABLE VENDOR DATASET (10 veg categories) ---
    veg_items = {
        # base_qty ~ 20–50 (kg/bundles per day for a local vendor)
        "Onion":       {"base_qty": 35, "base_price": 30, "type": "Vegetable"},
        "Potato":      {"base_qty": 40, "base_price": 25, "type": "Vegetable"},
        "Tomato":      {"base_qty": 32, "base_price": 40, "type": "Vegetable"},
        "Cabbage":     {"base_qty": 22, "base_price": 25, "type": "Vegetable"},
        "Cauliflower": {"base_qty": 20, "base_price": 30, "type": "Vegetable"},
        "Spinach":     {"base_qty": 28, "base_price": 15, "type": "Vegetable"},
        "Brinjal":     {"base_qty": 24, "base_price": 35, "type": "Vegetable"},
        "LadyFinger":  {"base_qty": 26, "base_price": 30, "type": "Vegetable"},
        "Capsicum":    {"base_qty": 18, "base_price": 50, "type": "Vegetable"},
        "GreenChilli": {"base_qty": 15, "base_price": 60, "type": "Vegetable"},
    }

    veg_in_season = {
        # In-season assumptions for some veg
        "Onion":       [11, 12, 1, 2],
        "Potato":      [11, 12, 1, 2],
        "Tomato":      [3, 4, 5, 6],
        "Spinach":     [11, 12, 1, 2],
        "Cauliflower": [11, 12, 1, 2],
    }

    generate_dataset(
        items=veg_items,
        veg_in_season=veg_in_season,
        output_path="Raw_Data/vegetable_sales_raw.csv",
    )


if __name__ == "__main__":
    main()

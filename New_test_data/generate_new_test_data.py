import pandas as pd
from datetime import datetime, timedelta
import random

def month_to_season(m):
    if m in [11, 12, 1, 2]:
        return "Winter"
    elif m in [3, 4, 5, 6]:
        return "Summer"
    elif m in [7, 8, 9]:
        return "Rainy"
    else:
        return "Transition"


def generate_new_test_data(kind="fruit", days=3, output_path="new_test_data.csv"):
    if kind == "fruit":
        items = ["Apple", "Banana", "Grapes", "Mango", "Orange"]
    else:
        items = ["Onion", "Potato", "Tomato", "Spinach", "Cabbage",
                 "Brinjal", "Cauliflower", "Capsicum", "LadyFinger", "GreenChilli"]

    today = datetime(2025, 12, 1)  # fixed date for reproducibility
    rows = []

    for i in range(days):
        date = today + timedelta(days=i)
        for item in items:
            price = random.randint(20, 150)  # random price
            weather = random.choice(["Sunny", "Cloudy", "Rain"])
            month = date.month

            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "item": item,
                "price": price,
                "weather": weather,
                "is_event": "No",
                "is_weekend": 1 if date.weekday() >= 5 else 0,
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"✅ New test data saved → {output_path}")
    print(df.head())


if __name__ == "__main__":
    # generate fruit test file
    generate_new_test_data(kind="fruit", days=3, output_path="new_test_fruit_data.csv")

    # generate vegetable test file
    generate_new_test_data(kind="veg", days=3, output_path="new_test_veg_data.csv")

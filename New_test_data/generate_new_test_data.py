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


def generate_new_test_data(kind="fruit", days=7, output_path="new_test_data.csv"):
    if kind == "fruit":
        items = ["Apple", "Banana", "Grapes", "Mango", "Orange"]
    else:
        items = ["Onion", "Potato", "Tomato", "Spinach", "Cabbage",
                 "Brinjal", "Cauliflower", "Capsicum", "LadyFinger", "GreenChilli"]

    today = datetime(2025, 12, 1)  # fixed date for reproducibility

    # --------------------------
    # ADD 2 RANDOM EVENT DAYS
    # --------------------------
    event_indices = random.sample(range(days), 2)  # choose 2 unique event days

    rows = []

    for i in range(days):
        date = today + timedelta(days=i)

        # Mark event if this index is selected
        is_event = "Yes" if i in event_indices else "No"

        for item in items:
            price = random.randint(20, 150)
            weather = random.choice(["Sunny", "Cloudy", "Rain"])
            month = date.month

            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "item": item,
                "price": price,
                "weather": weather,
                "is_event": is_event,               # <-- EVENT FLAG
                "is_weekend": 1 if date.weekday() >= 5 else 0,
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"✅ New test data saved → {output_path}")
    print(df.head())


if __name__ == "__main__":
    generate_new_test_data(kind="fruit", days=7, output_path="new_test_fruit_data.csv")
    generate_new_test_data(kind="veg", days=7, output_path="new_test_veg_data.csv")

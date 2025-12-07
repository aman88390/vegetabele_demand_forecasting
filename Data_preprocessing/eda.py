import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")


def load_data(path: str) -> pd.DataFrame:
    """Load a cleaned CSV and parse dates."""
    print(f"\n📂 Loading: {path}")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    print(df.head())
    return df


def eda_basic(df: pd.DataFrame, title: str = "Dataset"):
    """Print basic info, missing values, and distributions."""
    print(f"\n========== 📊 BASIC EDA: {title} ==========")

    print("\n🔹 Shape:", df.shape)

    print("\n🔍 Missing values:")
    print(df.isna().sum())

    print("\n🧺 Items:")
    print(df["item"].value_counts())

    print("\n🌦 Weather distribution:")
    print(df["weather"].value_counts())

    print("\n📅 Season distribution:")
    print(df["season"].value_counts())


def plot_trends(df: pd.DataFrame, title_prefix: str = ""):
    """High-level plots: total demand over time, price vs quantity, season, weather, weekend."""
    # Total quantity over time
    plt.figure(figsize=(12, 4))
    daily = df.groupby("date")["quantity_sold"].sum()
    daily.plot()
    plt.title(f"{title_prefix} Total Quantity Sold Over Time")
    plt.xlabel("Date")
    plt.ylabel("Quantity Sold")
    plt.tight_layout()
    plt.show()

    # Price vs quantity
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x="price", y="quantity_sold", alpha=0.6)
    plt.title(f"{title_prefix} Price vs Quantity Sold")
    plt.tight_layout()
    plt.show()

    # Seasonality
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="season", y="quantity_sold")
    plt.title(f"{title_prefix} Quantity Sold by Season")
    plt.tight_layout()
    plt.show()

    # Weather
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="weather", y="quantity_sold")
    plt.title(f"{title_prefix} Weather Impact on Demand")
    plt.tight_layout()
    plt.show()

    # Weekend vs weekday
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="is_weekend", y="quantity_sold")
    plt.title(f"{title_prefix} Weekend vs Weekday Demand")
    plt.xlabel("is_weekend (0=Weekday, 1=Weekend)")
    plt.tight_layout()
    plt.show()


def item_wise_eda(df: pd.DataFrame, title_prefix: str = ""):
    """Plot time series and price–demand for each item separately."""
    items = df["item"].unique()

    for item in items:
        item_df = df[df["item"] == item]

        # Time series
        plt.figure(figsize=(12, 4))
        plt.plot(item_df["date"], item_df["quantity_sold"])
        plt.title(f"{title_prefix} {item} – Demand Over Time")
        plt.xlabel("Date")
        plt.ylabel("Quantity Sold")
        plt.tight_layout()
        plt.show()

        # Price vs Demand
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=item_df, x="price", y="quantity_sold")
        plt.title(f"{title_prefix} {item} – Price vs Quantity Sold")
        plt.tight_layout()
        plt.show()

def _prepare_corr_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a numeric-only dataframe for correlation:
    encodes season & weather as codes and selects relevant columns.
    """
    corr_df = df.copy()

    # Encode categorical season & weather
    corr_df["season_code"] = corr_df["season"].astype("category").cat.codes
    corr_df["weather_code"] = corr_df["weather"].astype("category").cat.codes

    num_cols = [
        "quantity_sold",
        "price",
        "is_event",
        "is_weekend",
        "day_of_week",
        "month",
        "season_code",
        "weather_code",
    ]

    # Keep only columns that actually exist (safety)
    num_cols = [c for c in num_cols if c in corr_df.columns]

    return corr_df[num_cols]


def plot_correlation_heatmap(df: pd.DataFrame, title_prefix: str = ""):
    """
    Plot a correlation heatmap for key numeric + encoded categorical features.
    Use separately for fruits and vegetables.
    """
    corr_df = _prepare_corr_frame(df)
    corr_matrix = corr_df.corr()

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"{title_prefix} Correlation Heatmap")
    plt.tight_layout()
    plt.show()

def plot_item_correlation(df: pd.DataFrame, title_prefix: str = ""):
    """
    Correlation heatmap between items (fruits or vegetables).
    Shows which items' demand move together.
    """

    # Pivot: rows = date, columns = item, values = quantity_sold
    pivot_df = df.pivot_table(
        index="date",
        columns="item",
        values="quantity_sold",
        aggfunc="sum"
    )

    # Compute correlation between items
    corr_matrix = pivot_df.corr()

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"{title_prefix} Item-to-Item Demand Correlation")
    plt.tight_layout()
    plt.show()


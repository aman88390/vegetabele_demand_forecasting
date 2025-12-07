# 🥕 Street Vendor Demand Forecasting

A complete end-to-end demand prediction system for fruits & vegetables street vendors.

---

## 🚀 Project Overview

This project builds a demand prediction module for a street-vendor setup using historical daily sales data of fruits and vegetables.

**Key Features:**
- Realistic synthetic dataset generation with real-world vendor behavior patterns
- Comprehensive data cleaning, preprocessing, and feature engineering
- Exploratory Data Analysis (EDA) & visualization
- Multiple modeling experiments with best-model selection
- 7-day ahead demand forecasting
- **Three integration modes:**
  - ✔ Python script
  - ✔ Web Dashboard (Streamlit)
  - ✔ REST API (Flask)

---

## 📂 Directory Structure

```
VEGETBLE_DEMAND_FORECASTING/
│
├── 7_days_prediction/
│   ├── fruit_7day_predictions.csv
│   └── veg_7day_predictions.csv
│
├── Data_preprocessing/
│   ├── data_preprocessing.py
│   ├── prepare_features.py
│   ├── fruit_sales_clean.csv
│   ├── vegetable_sales_clean.csv
│   ├── eda.py
│   └── visualize.ipynb
│
├── Model_Data/
│   ├── final_model_metrics.csv
│   ├── final_models/
│   │   ├── fruit_itemwise_linear_regression.pkl
│   │   └── veg_global_random_forest.pkl
│   ├── fruit_encoder.pkl
│   ├── fruit_scaler.pkl
│   ├── fruit_feature_cols.json
│   ├── veg_encoder.pkl
│   ├── veg_scaler.pkl
│   └── veg_feature_cols.json
│
├── New_test_data/
│   ├── generate_new_test_data.py
│   ├── new_test_fruit_data.csv
│   └── new_test_veg_data.csv
│
├── Raw_Data/
│   ├── fruit_sales_raw.csv
│   ├── vegetable_sales_raw.csv
│   └── generate_data.py
│
├── model/
│   ├── train_models.py
│   ├── predict_next_7_days.py
│   └── modes_experiments.ipynb
│
├── dashboard.py
├── demand_api.py
├── requirements.txt
└── run_all.py
```

---

## 🍉 Data Generation & Assumptions

Synthetic data was generated to mimic real-world vendor behavior with the following assumptions:

### A. Vendor Type Separation
- Vendors typically sell **only fruits** OR **only vegetables**
- Two separate datasets: `fruit_sales_raw.csv` (5 items) and `vegetable_sales_raw.csv` (10 items)

### B. Pricing & Quantities
- **Fruits:** Higher base prices (₹80–₹150)
- **Vegetables:** Mid-range/low prices (₹20–₹60)
- **Quantities:** Realistic for street vendors (5–40 kg/day)

### C. Seasonality
- **Winter (Nov–Feb):** Onions/potatoes cheap & abundant
- **Summer (Mar–Jun):** Tomatoes abundant, higher fruit demand
- **Rainy (Jul–Sep):** Lower vegetable supply, higher prices
- **Transition (Oct):** Mixed effects

### D. Weekend Effect
- Increased demand on weekends for family purchases
- Fruits see higher weekend demand than vegetables

### E. Event Days (Festivals/Weddings)
- ~15 random event days with increased demand
- Vendors raise prices during events
- Weather has minimal impact on event-day demand

### F. Weather Effects
- **Sunny:** Slight demand increase
- **Cloudy:** Neutral
- **Rain:** 40-50% demand drop on normal days

### G. Realistic Noise & Outliers
- Missing values (3%)
- Invalid entries (negative quantities, outliers)
- Duplicate rows (2%)
- Random bulk orders and supply issues

### H. Data Range
All data ends before **December 31, 2025**

---

## 🧹 Data Processing Pipeline

### 1. Raw Data Generation
```bash
python Raw_Data/generate_data.py
or 
python3 Raw_Data/generate_data.py
```
Creates: `fruit_sales_raw.csv` and `vegetable_sales_raw.csv`

### 2. Cleaning & Preprocessing
```bash
python Data_preprocessing/data_preprocessing.py
or 
python3 Data_preprocessing/data_preprocessing.py
```
**Tasks:**
- Parse and validate dates
- Remove duplicates
- Handle missing values
- Fix outliers
- Clean event flags and weather categories
- Add calendar features (month, season)

**Outputs:** `fruit_sales_clean.csv` and `vegetable_sales_clean.csv`

### 3. Feature Preparation
```bash
python Data_preprocessing/prepare_features.py
or 
python3 Data_preprocessing/prepare_features.py
```
**Steps:**
- One-hot encode: item, weather, season
- Numeric features: price, is_event, is_weekend, day_of_week, month
- Standardize with StandardScaler

**Outputs:** Feature matrices, encoders, scalers, and feature column definitions

### 4. EDA & Visualization
```bash
python Data_preprocessing/eda.py
or 
python3 Data_preprocessinf/eda.py
```
Or explore interactively: `Data_preprocessing/visualize.ipynb`

**Includes:**
- Demand trends over time
- Price vs demand analysis
- Seasonal and weather effects
- Item-wise patterns
- Correlation heatmaps

---

## 🤖 Modeling & Experiments

All experiments documented in: `model/modes_experiments.ipynb`

### Models Evaluated

**Baseline:**
- Item-wise 7-day Moving Average

**Global ML Models:**
- Linear Regression (single model for all items)
- Random Forest Regressor (single model for all items)

**Item-wise Models:**
- Linear Regression per item (Item_LR)

**Time-series Models:**
- Item-wise ARIMA (2,1,2)

All models use chronological 80/20 train-test splits.

---

## 🏆 Best Model Selection

### Fruits: Item-wise Linear Regression
- Separate Linear Regression per fruit
- Best MAE and R² per item
- Excellent interpretability and stability
- **Saved as:** `Model_Data/final_models/fruit_itemwise_linear_regression.pkl`

### Vegetables: Global Random Forest
- Single Random Forest across all vegetables
- Handles non-linear interactions, price effects, and seasonality
- Best MAE (~6.7) and R² (~0.65)
- **Saved as:** `Model_Data/final_models/veg_global_random_forest.pkl`

---

## 🔮 7-Day Ahead Forecasting

### Python Script Mode
```bash
python model/predict_next_7_days.py
or 
python3 model/predict_next_7_days.py
```

**Process:**
1. Loads clean historical data
2. Generates future 7-day feature rows per item
3. Uses last observed prices and common weather patterns
4. Applies trained encoders and scalers
5. Predicts using best models

**Outputs:**
- `7_days_prediction/fruit_7day_predictions.csv`
- `7_days_prediction/veg_7day_predictions.csv`

---

## 📊 Web Dashboard (Streamlit)

### Launch Dashboard
```bash
streamlit run dashboard.py
```

### Features

**Tab 1: Existing 7-day Forecast**
- View pre-generated 7-day predictions
- Interactive line charts
- Tabular data display

**Tab 2: Predict from CSV**
- Upload custom CSV files
- Real-time predictions using best models
- Download results as CSV
- JSON-like tabular output

---

## 🌐 REST API (Flask)

### Launch API
```bash
python demand_api.py 
or
python3 demand_api.py
```

### API Endpoints

#### Predict Fruit Demand
```bash
curl -X POST "http://localhost:8000/predict/fruit" \
  -F "file=@new_test_fruit_data.csv"
```

#### Predict Vegetable Demand
```bash
curl -X POST "http://localhost:8000/predict/veg" \
  -F "file=@new_test_veg_data.csv"
```

**Returns:** JSON output with predicted demand

---

## 📥 Generate Test Data

Create your own test datasets:
```bash
python New_test_data/generate_new_test_data.py
```

**Generates:**
- `new_test_fruit_data.csv`
- `new_test_veg_data.csv`

These files can be uploaded to the Streamlit dashboard or API.

---

## 🛠 Tech Stack

- **Core:** Python, Pandas, NumPy
- **ML:** Scikit-learn, Statsmodels (ARIMA)
- **Visualization:** Matplotlib, Seaborn
- **Web Dashboard:** Streamlit
- **REST API:** Flask
- **Model Persistence:** Joblib

---

## 🧠 How It Works (Plain English)

1. **Clean the data:** Fix dates, remove duplicates, handle missing values and outliers
2. **Engineer features:** Create meaningful inputs like price, weekday/weekend, season, weather, events
3. **Learn patterns:** Train models to understand relationships between features and demand
4. **Select best models:**
   - Fruits: Item-wise approach works best (each fruit has unique patterns)
   - Vegetables: Global approach works best (shared non-linear patterns)
5. **Forecast 7 days ahead:**
   - Extend dates beyond last known day
   - Use last observed prices and common weather
   - Predict demand per item per day

---

## ▶️ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline
```bash
python run_all.py
or
python3 run_all.py
```
This executes: data generation → cleaning → feature engineering → model training → 7-day forecasting

### 3. View Predictions
```bash
ls 7_days_prediction/
```

### 4. Launch Dashboard
```bash
streamlit run dashboard.py
```

### 5. Launch REST API
```bash
python demand_api.py
or
python3 demand_api.py
```

---

## 📈 Model Performance

Results available in: `Model_Data/final_model_metrics.csv`

**Fruits (Item-wise Linear Regression):**
- Per-item optimization
- High interpretability
- Stable predictions

**Vegetables (Global Random Forest):**
- MAE: ~6.7 kg
- R²: ~0.65
- Handles complex interactions

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

---

## 📝 License

This project is open source and available for educational and commercial use.

---

## 👥 Contact

For questions or feedback, please open an issue in the repository.

---

**Built with ❤️ for street vendors everywhere**
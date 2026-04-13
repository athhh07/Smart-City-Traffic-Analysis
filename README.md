# Smart City Traffic Prediction 🚦

Predicting hourly vehicle counts at 4 urban junctions using Machine Learning and Time Series models.

Built as a Data Science internship/college project on the [Kaggle Smart City Traffic Patterns dataset](https://www.kaggle.com/).

---

## Project Overview

Traffic congestion is a major problem in smart cities. This project builds a prediction system that forecasts how many vehicles will pass through 4 different junctions every hour — helping city planners make better decisions about traffic signal timing and road management.

---

## Dataset

| File | Description |
|---|---|
| `train_aWnotu.csv` | Historical traffic data with vehicle counts |
| `test.csv` | Future timestamps to predict |

**Columns:**
- `DateTime` — hourly timestamp
- `Junction` — junction number (1, 2, 3, 4)
- `Vehicles` — number of vehicles (target variable, only in train)
- `ID` — unique row identifier

---

## Approach

### Models Used

| Model | Why |
|---|---|
| **XGBoost** | Powerful tree-based model, great at learning from features like hour, day, weekend |
| **SARIMA** | Time series model that captures weekly traffic patterns (Mon–Sun cycle) |
| **Blended (final)** | 60% XGBoost + 40% SARIMA — combines strengths of both models |

### Key Design Decisions

- **Time-based train/val split** — last 4 weeks used for validation. Never random split on time series data (causes data leakage)
- **Separate SARIMA per junction** — each junction has its own traffic pattern
- **Blending** — combining two different model types almost always improves accuracy

---

## Features Created

| Feature | Description |
|---|---|
| `hour` | Hour of day (0–23) |
| `day` | Day of month |
| `month` | Month of year |
| `year` | Year |
| `day_of_week` | 0 = Monday, 6 = Sunday |
| `is_weekend` | 1 if Saturday or Sunday, else 0 |
| `is_peak` | 1 if rush hour (7–9am or 5–7pm), else 0 |

---

## Results

| Model | Validation RMSE |
|---|---|
| XGBoost | **6.56** |
| SARIMA | **16.80** |
| **Blended** | **9.13** |

> Lower RMSE = better predictions

---

## Project Structure

```
smart-city-traffic/
│
├── data/
│ └── raw/
│ ├── train_aWnotu.csv
│ └── test.csv
│
├── notebooks/
│ └── traffic_analysis.ipynb
│
├── src/
│ ├── preprocessing.py
│ ├── feature_engineering.py
│ ├── train_xgboost.py
│ ├── train_sarima.py
│ └── predict.py
│
├── models/
│ ├── xgb_model.pkl
│ └── sarima_models/
│ ├── junction_1.pkl
│ ├── junction_2.pkl
│ ├── junction_3.pkl
│ └── junction_4.pkl
│
├── outputs/
│ ├── plots/
│ │ ├── traffic_over_time.png
│ │ ├── hourly_trend.png
│ │ ├── weekday_vs_weekend.png
│ │ └── feature_importance.png
│ │
│ └── logs/
│ └── training_logs.txt
│
├── submissions/
│ └── submission.csv
│
├── main.py
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn xgboost scikit-learn pmdarima joblib
```

### 2. Run the notebook (recommended)

Open `notebooks/traffic_analysis.ipynb` in Jupyter and run cells top to bottom.

### 3. Or run the full pipeline from terminal

```bash
python main.py
```

---

## Notebook Walkthrough

| Step | What it does |
|---|---|
| 1. Import libraries | Load all required packages |
| 2. Load data | Read train and test CSV files |
| 3. Clean data | Convert DateTime, sort by junction and time |
| 4. EDA | Plot traffic patterns by hour, day, month |
| 5. Feature engineering | Extract hour, weekend flag, peak hour flag etc. |
| 6. XGBoost | Train and evaluate with time-based validation |
| 7. SARIMA | Train one model per junction (saves to disk) |
| 8. Blend | Combine both models, generate final predictions |
| 9. Save | Export submission.csv |

---

## Sample Visualizations

### Traffic over time (all junctions)
Shows how vehicle count changes across the full dataset period for each junction.

### Hourly average traffic
Shows the classic rush hour peaks at 7–9am and 5–7pm.

### Weekday vs weekend
Weekdays have sharp peaks; weekends have flatter, lower traffic throughout the day.

### Actual vs Predicted (XGBoost)
Orange dashed line closely follows the blue actual line — showing the model learned the daily traffic pattern well.

---

## Key Learnings

- **Never random-split time series data** — always split by time to avoid leakage
- **SARIMA needs a regular time index** — missing hours must be handled before fitting
- **Blending two models** with different strengths gives better results than either alone
- **Rush hour and weekends** are the strongest signals in traffic data

---

## Possible Improvements

- Add weather data (rain reduces traffic significantly)
- Add public holiday calendar
- Try Facebook Prophet (handles holidays natively)
- Try LightGBM (faster than XGBoost)
- Train separate XGBoost models per junction

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-green)
![SARIMA](https://img.shields.io/badge/SARIMA-pmdarima-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

- Python 3.10
- pandas, numpy
- XGBoost
- pmdarima (auto_arima)
- scikit-learn
- matplotlib, seaborn
- joblib

---

## Author : Atharva Desai 

Made as part of a Data Science internship @uoSkillCampus

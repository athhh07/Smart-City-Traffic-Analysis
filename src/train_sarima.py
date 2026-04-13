import pandas as pd
import numpy as np
import joblib
import os
 
from pmdarima import auto_arima   # pip install pmdarima
 
from preprocessing import load_data, clean_data
 
 
def train_sarima_for_junction(train, junction_number):
    print(f"\nTraining SARIMA for Junction {junction_number}...")
 
    junction_data = train[train['Junction'] == junction_number]
 
    series = junction_data.set_index('DateTime')['Vehicles']
    model = auto_arima(
        series,
        seasonal          = True,
        m                 = 7,    # weekly pattern (was 24 — caused MemoryError)
        max_p             = 2,    # limit AR terms
        max_q             = 2,    # limit MA terms
        max_P             = 1,    # limit seasonal AR terms
        max_Q             = 1,    # limit seasonal MA terms
        stepwise          = True,
        suppress_warnings = True,
        error_action      = 'ignore'
    )
 
    print(f"  Best order found: {model.order}, seasonal: {model.seasonal_order}")
 
    return model
 
 
def train_all_junctions(train):
    os.makedirs('models/sarima_models', exist_ok=True)
 
    sarima_models = {}
 
    for junction in [1, 2, 3, 4]:
        model = train_sarima_for_junction(train, junction)
        sarima_models[junction] = model
 
        save_path = f'models/sarima_models/junction_{junction}.pkl'
        joblib.dump(model, save_path)
        print(f"  Saved to {save_path}")
 
    print("\nAll SARIMA models trained and saved!")
    return sarima_models
 
 
if __name__ == '__main__':
    train, test = load_data('data/raw/train_aWnotu.csv', 'data/raw/test.csv')
    train, test = clean_data(train, test)
 
    sarima_models = train_all_junctions(train)
 
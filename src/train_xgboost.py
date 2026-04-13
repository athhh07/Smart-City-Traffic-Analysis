import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

FEATURES = ['hour','day','month','year','day_of_week','is_weekend','is_peak','Junction']
TARGET = 'Vehicles'

def train_xgb(train):
    cutoff_date = train['DateTime'].max() - pd.Timedelta(weeks=4)
    train_data = train[train['DateTime'] <= cutoff_date]
    val_data = train[train['DateTime'] > cutoff_date]

    X_train = train_data[FEATURES]
    y_train = train_data[TARGET]
    X_val = val_data[FEATURES]
    y_val = val_data[TARGET]

    model = XGBRegressor(n_estimators=500,learning_rate=0.05,max_depth=5,random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    preds = np.maximum(preds, 0)
    rmse = np.sqrt(mean_squared_error(y_val, preds))

    return model, rmse, val_data
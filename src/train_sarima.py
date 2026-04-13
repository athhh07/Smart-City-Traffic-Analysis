import os
import joblib
import numpy as np
from pmdarima import auto_arima

def train_sarima(train):
    os.makedirs('sarima_models', exist_ok=True)
    models = {}

    for junction in [1,2,3,4]:
        path = f'sarima_models/junction_{junction}.pkl'
        if os.path.exists(path):
            models[junction] = joblib.load(path)
            continue

        data = train[train['Junction'] == junction]
        series = data.set_index('DateTime')['Vehicles']

        model = auto_arima(series,seasonal=True,m=7,max_p=2,max_q=2,max_P=1,max_Q=1,stepwise=True,suppress_warnings=True,error_action='ignore')
        joblib.dump(model, path)
        models[junction] = model

    return models
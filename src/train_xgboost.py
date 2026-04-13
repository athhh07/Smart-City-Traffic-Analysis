import pandas as pd
import numpy as np
import joblib
 
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
 
from preprocessing import load_data, clean_data
from feature_engineering import add_features, FEATURES
 
 
def train_xgboost(train):
 
    cutoff_date = train['DateTime'].max() - pd.Timedelta(weeks=4)
 
    train_data = train[train['DateTime'] <= cutoff_date]
    val_data   = train[train['DateTime'] >  cutoff_date]
 
    print("Training rows  :", len(train_data))
    print("Validation rows:", len(val_data))
    print("Cutoff date    :", cutoff_date.date())
 
    X_train = train_data[FEATURES]
    y_train = train_data['Vehicles']
 
    X_val   = val_data[FEATURES]
    y_val   = val_data['Vehicles']
 
    model = XGBRegressor(
        n_estimators  = 500,    # number of trees
        learning_rate = 0.05,   # how much each tree contributes
        max_depth     = 5,      # how deep each tree goes
        random_state  = 42
    )
 
    model.fit(X_train, y_train)
 
    predictions = model.predict(X_val)
 
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    print("\nValidation RMSE:", round(rmse, 2))
 
    joblib.dump(model, 'models/xgb_model.pkl')
    print("Model saved to models/xgb_model.pkl")
 
    return model
 
 
if __name__ == '__main__':
    train, test = load_data('data/raw/train_aWnotu.csv', 'data/raw/test.csv')
    train, test = clean_data(train, test)
 
    train = add_features(train)
    test  = add_features(test)
 
    model = train_xgboost(train)
 
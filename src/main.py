import os
import pandas as pd
from preprocessing import load_data, preprocess
from feature_engineering import add_features
from train_xgboost import train_xgb
from train_sarima import train_sarima
from predict import predict

train, test = load_data()
train, test = preprocess(train, test)

train = add_features(train)
test = add_features(test)

xgb_model, rmse, val_data = train_xgb(train)
sarima_models = train_sarima(train)

final_preds = predict(test, xgb_model, sarima_models)

os.makedirs('submissions', exist_ok=True)

submission = pd.DataFrame({
    'ID': test['ID'],
    'Vehicles': final_preds
})

submission.to_csv('submissions/submission.csv', index=False)
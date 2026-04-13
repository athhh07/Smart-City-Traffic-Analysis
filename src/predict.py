import pandas as pd
import numpy as np
import joblib
import os

from preprocessing import load_data, clean_data
from feature_engineering import add_features, FEATURES


def load_models():
    xgb_model = joblib.load('models/xgb_model.pkl')
    print("XGBoost model loaded.")

    sarima_models = {}
    for junction in [1, 2, 3, 4]:
        path = f'models/sarima_models/junction_{junction}.pkl'
        if os.path.exists(path):
            sarima_models[junction] = joblib.load(path)
            print(f"SARIMA Junction {junction} loaded.")
        else:
            print(f"WARNING: SARIMA model for Junction {junction} not found!")

    return xgb_model, sarima_models


def predict_xgboost(xgb_model, test):
    X_test = test[FEATURES]
    predictions = xgb_model.predict(X_test)
    predictions = np.maximum(predictions, 0)
    return predictions


def predict_sarima(sarima_models, test):
    sarima_predictions = []

    for junction in [1, 2, 3, 4]:
        junction_test = test[test['Junction'] == junction]
        n_periods = len(junction_test)

        preds = sarima_models[junction].predict(n_periods=n_periods)
        preds = np.maximum(preds, 0)

        sarima_predictions.append(
            pd.Series(preds, index=junction_test.index)
        )

    all_preds = pd.concat(sarima_predictions)
    all_preds = all_preds.sort_index()

    return all_preds.values


def blend_and_save(test, xgb_preds, sarima_preds):
    alpha = 0.6
    final_preds = alpha * xgb_preds + (1 - alpha) * sarima_preds
    final_preds = np.round(final_preds).astype(int)

    submission = pd.DataFrame({
        'ID': test['ID'],
        'Vehicles': final_preds
    })

    os.makedirs('outputs/submissions', exist_ok=True)
    submission.to_csv('outputs/submissions/submission.csv', index=False)

    print("\nSubmission saved to outputs/submissions/submission.csv")
    print(submission.head(10))

    return submission


if __name__ == '__main__':
    train, test = load_data('data/raw/train_aWnotu.csv', 'data/raw/test.csv')
    train, test = clean_data(train, test)
    test = add_features(test)

    xgb_model, sarima_models = load_models()

    print("\nGenerating XGBoost predictions...")
    xgb_preds = predict_xgboost(xgb_model, test)

    print("Generating SARIMA predictions...")
    sarima_preds = predict_sarima(sarima_models, test)

    submission = blend_and_save(test, xgb_preds, sarima_preds)
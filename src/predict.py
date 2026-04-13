import numpy as np
import pandas as pd

FEATURES = ['hour','day','month','year','day_of_week','is_weekend','is_peak','Junction']

def predict(test, xgb_model, sarima_models):
    X_test = test[FEATURES].fillna(0)
    xgb_preds = xgb_model.predict(X_test)
    xgb_preds = np.maximum(xgb_preds, 0)

    sarima_list = []

    for junction in [1,2,3,4]:
        jtest = test[test['Junction'] == junction]
        n = len(jtest)
        preds = sarima_models[junction].predict(n_periods=n)
        preds = np.array(preds, dtype=float)
        preds = np.maximum(preds, 0)
        sarima_list.append(pd.Series(preds, index=jtest.index))

    sarima_preds = pd.concat(sarima_list).sort_index()
    sarima_preds = sarima_preds.fillna(sarima_preds.mean()).values

    final_preds = 0.6 * xgb_preds + 0.4 * sarima_preds
    final_preds = np.round(final_preds).astype(int)

    return final_preds
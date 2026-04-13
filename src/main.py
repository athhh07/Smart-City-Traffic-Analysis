from preprocessing import load_data, clean_data
from feature_engineering import add_features
from train_xgboost import train_xgboost
from train_sarima import train_all_junctions
from predict import predict_xgboost, predict_sarima, blend_and_save


def main():
    print("=" * 50)
    print("STEP 1: Load and clean data")
    print("=" * 50)
    train, test = load_data('data/raw/train_aWnotu.csv', 'data/raw/test.csv')
    train, test = clean_data(train, test)

    print("\n" + "=" * 50)
    print("STEP 2: Feature engineering")
    print("=" * 50)
    train = add_features(train)
    test = add_features(test)

    print("\n" + "=" * 50)
    print("STEP 3: Train XGBoost")
    print("=" * 50)
    xgb_model = train_xgboost(train)

    print("\n" + "=" * 50)
    print("STEP 4: Train SARIMA (takes ~10-30 min)")
    print("=" * 50)
    sarima_models = train_all_junctions(train)

    print("\n" + "=" * 50)
    print("STEP 5: Generate predictions and save")
    print("=" * 50)
    xgb_preds = predict_xgboost(xgb_model, test)
    sarima_preds = predict_sarima(sarima_models, test)
    blend_and_save(test, xgb_preds, sarima_preds)

    print("\nDone! Check outputs/submissions/submission.csv")


if __name__ == '__main__':
    main()
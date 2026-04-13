import pandas as pd

def load_data():
    train = pd.read_csv('data/raw/train_aWnotu.csv')
    test = pd.read_csv('data/raw/test.csv')
    return train, test

def preprocess(train, test):
    train['DateTime'] = pd.to_datetime(train['DateTime'])
    test['DateTime'] = pd.to_datetime(test['DateTime'])
    train = train.sort_values(['Junction', 'DateTime']).reset_index(drop=True)
    test = test.sort_values(['Junction', 'DateTime']).reset_index(drop=True)
    return train, test
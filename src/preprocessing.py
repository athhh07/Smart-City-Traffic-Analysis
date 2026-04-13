
 
import pandas as pd
 
 
def load_data(train_path, test_path):    
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
 
    print("Train shape:", train.shape)
    print("Test shape :", test.shape)
 
    return train, test
 
 
def clean_data(train, test):
    train['DateTime'] = pd.to_datetime(train['DateTime'])
    test['DateTime']  = pd.to_datetime(test['DateTime'])
 
    train = train.sort_values(['Junction', 'DateTime'])
    test  = test.sort_values(['Junction', 'DateTime'])
 
    train = train.reset_index(drop=True)
    test  = test.reset_index(drop=True)
 
    print("\nMissing values in train:")
    print(train.isnull().sum())
 
    return train, test
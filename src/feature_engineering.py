import pandas as pd
 
 
def add_features(df):
 
    df['hour']        = df['DateTime'].dt.hour         # 0 to 23
    df['day']         = df['DateTime'].dt.day           # 1 to 31
    df['month']       = df['DateTime'].dt.month         # 1 to 12
    df['year']        = df['DateTime'].dt.year          # e.g. 2015
    df['day_of_week'] = df['DateTime'].dt.dayofweek    # 0=Monday, 6=Sunday
 
    df['is_weekend'] = 0
    df.loc[df['day_of_week'] >= 5, 'is_weekend'] = 1
 
    peak_hours = [7, 8, 9, 17, 18, 19]
    df['is_peak'] = 0
    df.loc[df['hour'].isin(peak_hours), 'is_peak'] = 1
 
    return df
 
FEATURES = [
    'hour',
    'day',
    'month',
    'year',
    'day_of_week',
    'is_weekend',
    'is_peak',
    'Junction'
]
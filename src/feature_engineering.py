def add_features(df):
    
    df = df.copy()
    df['hour'] = df['DateTime'].dt.hour
    df['day'] = df['DateTime'].dt.day
    df['month'] = df['DateTime'].dt.month
    df['year'] = df['DateTime'].dt.year
    df['day_of_week'] = df['DateTime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    peak_hours = [7, 8, 9, 17, 18, 19]
    df['is_peak'] = df['hour'].isin(peak_hours).astype(int)
    
    return df
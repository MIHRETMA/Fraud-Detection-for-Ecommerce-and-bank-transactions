import pandas as pd

def cleaning(df):
    df.drop_duplicates(inplace=True) # drop duplicates
    
    # Convert time-related columns to datetime format if they exist
    if 'signup_time' in df.columns:
        df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce')
    if 'purchase_time' in df.columns:
        df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')

    return df
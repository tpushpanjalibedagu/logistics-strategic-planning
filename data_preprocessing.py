import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def run_preprocessing_pipeline(file_path: str):
    """
    Loads, cleans, handles outliers, and normalizes logistics dataset features.
    """
    df = pd.read_csv(file_path)
    
    # Step 1: Handle Missing Data
    df['package_weight_kg'] = df['package_weight_kg'].fillna(df['package_weight_kg'].median())
    df['actual_delivery_min'] = df['actual_delivery_min'].fillna(df['planned_delivery_min'])
    
    # Step 2: Coordinate Bounds Check
    df = df[(df['latitude'].between(-90, 90)) & (df['longitude'].between(-180, 180))].copy()
    
    # Step 3: Outlier Capping (IQR)
    Q1 = df['actual_delivery_min'].quantile(0.25)
    Q3 = df['actual_delivery_min'].quantile(0.75)
    IQR = Q3 - Q1
    df['actual_delivery_min'] = np.clip(df['actual_delivery_min'], Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    
    # Step 4: Normalization
    scaler = MinMaxScaler()
    df[['norm_weight', 'norm_planned_time']] = scaler.fit_transform(
        df[['package_weight_kg', 'planned_delivery_min']]
    )
    
    return df

if __name__ == "__main__":
    print("Preprocessing pipeline loaded successfully.")
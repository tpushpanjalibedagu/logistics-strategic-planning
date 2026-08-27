import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def build_logistics_pipeline(data_path: str):
    """
    Ingests raw delivery manifest data, cleans coordinate records,
    and applies K-Means clustering to partition geographic zones.
    """
    # 1. Load Data
    df = pd.read_csv(data_path)
    df['order_timestamp'] = pd.to_datetime(df['order_timestamp'])
    df['hour_of_day'] = df['order_timestamp'].dt.hour
    
    # 2. Clean Invalid Coordinate Bounds
    clean_df = df[(df['latitude'].between(-90, 90)) & 
                 (df['longitude'].between(-180, 180))].copy()

    # 3. Spatial Clustering for Delivery Zones
    coords = clean_df[['latitude', 'longitude']].values
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clean_df['delivery_zone'] = kmeans.fit_predict(coords)

    # 4. Distance to Zone Centroid Proxy (Km)
    centroids = kmeans.cluster_centers_
    clean_df['dist_to_depot_km'] = np.min(
        cdist(coords, centroids, metric='euclidean') * 111, axis=1
    )

    return clean_df, centroids

if __name__ == "__main__":
    print("Logistics Data Exploration Framework initialized successfully.")
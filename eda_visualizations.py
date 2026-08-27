import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_logistics_visualizations():
    # 1. Synthesize Dataset for Analysis
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'distance_km': np.random.gamma(shape=3, scale=4, size=n),
        'shipment_volume_m3': np.random.uniform(0.1, 4.0, size=n),
        'vehicle_type': np.random.choice(['Van', 'Cargo E-Bike', 'Motorcycle'], size=n, p=[0.5, 0.3, 0.2])
    })
    
    # Deriving dependent target variables
    df['actual_delivery_min'] = df['distance_km'] * 2.5 + np.random.normal(10, 5, n)
    df['transport_cost_usd'] = df['distance_km'] * 1.8 + df['shipment_volume_m3'] * 12 + np.random.normal(5, 2, n)
    df['delay_minutes'] = df['actual_delivery_min'] - (df['distance_km'] * 2.0 + 15)

    # Set visual theme
    sns.set_theme(style="whitegrid")

    # Chart 1: SLA Delay Distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(df['delay_minutes'], kde=True, color='crimson', bins=30)
    plt.title('Distribution of Delivery Delays (SLA Variance)', fontsize=12, fontweight='bold')
    plt.xlabel('Delay (Minutes)')
    plt.ylabel('Order Frequency')
    plt.tight_layout()
    plt.savefig('delay_distribution.png')
    plt.close()

    # Chart 2: Transit Distance vs Duration Regression
    plt.figure(figsize=(8, 5))
    sns.regplot(data=df, x='distance_km', y='actual_delivery_min',
                scatter_kws={'alpha':0.4, 'color':'navy'}, line_kws={'color':'red'})
    plt.title('Trip Distance vs. Actual Delivery Time', fontsize=12, fontweight='bold')
    plt.xlabel('Distance (km)')
    plt.ylabel('Delivery Duration (mins)')
    plt.tight_layout()
    plt.savefig('distance_vs_time.png')
    plt.close()

    # Chart 3: Cost Structure by Vehicle Type
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x='vehicle_type', y='transport_cost_usd', palette='Set2')
    plt.title('Transport Cost Distribution by Vehicle Mode', fontsize=12, fontweight='bold')
    plt.xlabel('Vehicle Type')
    plt.ylabel('Transport Cost ($)')
    plt.tight_layout()
    plt.savefig('cost_by_vehicle.png')
    plt.close()

    print("Visualizations successfully generated and saved to directory.")

if __name__ == "__main__":
    generate_logistics_visualizations()
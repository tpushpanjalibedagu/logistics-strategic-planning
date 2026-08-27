import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_predictive_pipeline():
    # 1. Synthesize Dataset
    np.random.seed(42)
    n = 1000
    distance_km = np.random.uniform(1.0, 35.0, n)
    shipment_volume_m3 = np.random.uniform(0.1, 4.0, n)
    package_weight_kg = np.random.uniform(0.5, 50.0, n)
    hour_of_day = np.random.randint(8, 20, n)
    is_peak_hour = np.where((hour_of_day in [8, 9, 10, 17, 18, 19]), 1, 0)
    
    # Target variable generation with synthetic non-linear noise
    actual_delivery_min = (
        distance_km * 2.2 + 
        is_peak_hour * 14.5 + 
        package_weight_kg * 0.15 + 
        np.random.normal(5, 3, n)
    )
    
    df = pd.DataFrame({
        'distance_km': distance_km,
        'shipment_volume_m3': shipment_volume_m3,
        'package_weight_kg': package_weight_kg,
        'hour_of_day': hour_of_day,
        'is_peak_hour': is_peak_hour,
        'actual_delivery_min': actual_delivery_min
    })

    # 2. Features and Target Split
    X = df.drop(columns=['actual_delivery_min'])
    y = df['actual_delivery_min']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Baseline Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)

    # 4. Hyperparameter Tuning for Random Forest
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    rf_preds = best_rf.predict(X_test)

    # 5. Model Evaluation
    print("=== Model Performance Results ===")
    print(f"Linear Regression -> MAE: {mean_absolute_error(y_test, lr_preds):.2f} | R2: {r2_score(y_test, lr_preds):.4f}")
    print(f"Random Forest (Tuned) -> MAE: {mean_absolute_error(y_test, rf_preds):.2f} | R2: {r2_score(y_test, rf_preds):.4f}")
    print(f"Random Forest RMSE: {np.sqrt(mean_squared_error(y_test, rf_preds)):.2f}")

    # Feature Importance
    importances = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n=== Feature Importances ===")
    print(importances)

    return best_rf

if __name__ == "__main__":
    run_predictive_pipeline()
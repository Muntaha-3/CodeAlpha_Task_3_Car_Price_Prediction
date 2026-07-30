import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load Dataset
df = pd.read_csv('car data.csv')

# 2. Feature Engineering
df['Car_Age'] = 2026 - df['Year']
df.drop(['Year', 'Car_Name'], axis=1, inplace=True)
df = pd.get_dummies(df, drop_first=True)

# 3. Separate Features (X) and Target Variable (y)
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# 4. Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model Training (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluation
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"--- MODEL EVALUATION RESULTS ---")
print(f"R² Score: {r2:.4f} (Accuracy around {r2*100:.1f}%)")
print(f"Mean Absolute Error: ${mae:.2f}k")

# 7. Plotting Actual vs Predicted Prices
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual Prices vs Predicted Prices')
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price')
plt.grid(True)
plt.show()

# --- 8. TEST WITH CUSTOM SAMPLE CAR ---
# Example: 5-year-old Petrol Manual car, 30,000 kms driven, present showroom price $8.5k
sample_car = pd.DataFrame([{
    'Present_Price': 8.5,
    'Driven_kms': 30000,
    'Owner': 0,
    'Car_Age': 5,
    'Fuel_Type_Diesel': False,
    'Fuel_Type_Petrol': True,
    'Selling_type_Individual': False,
    'Transmission_Manual': True
}])

predicted_price = model.predict(sample_car)[0]
print(f"\n🚘 Estimated Selling Price for Sample Car: ${predicted_price:.2f}k")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib


data = pd.read_csv('house_data.csv')

data.columns = data.columns.str.strip()

data = data.fillna(0)  # Simple approach, fill missing values with 0

# Encode categorical variables (SUBURB, NEAREST_STN, NEAREST_SCH)
label_encoders = {}
categorical_columns = ['SUBURB', 'NEAREST_STN', 'NEAREST_SCH']

for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Feature selection: use relevant features for prediction
X = data[['BEDROOMS', 'BATHROOMS', 'GARAGE', 'LAND_AREA', 'FLOOR_AREA', 
          'BUILD_YEAR', 'CBD_DIST', 'NEAREST_STN_DIST', 'LATITUDE', 'LONGITUDE', 
          'NEAREST_SCH_DIST', 'NEAREST_SCH_RANK']]

# Target variable is 'PRICE'
y = data['PRICE']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=500, max_depth=20, min_samples_split=5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Root Mean Squared Error: {rmse}")

joblib.dump(model, 'house_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

loaded_model = joblib.load('house_price_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

new_data = pd.DataFrame([[3, 2, 1, 450, 120, 1995, 12000, 1500, -32.1159, 115.8425, 0.8, 100]],
                        columns=['BEDROOMS', 'BATHROOMS', 'GARAGE', 'LAND_AREA', 'FLOOR_AREA', 
                                 'BUILD_YEAR', 'CBD_DIST', 'NEAREST_STN_DIST', 'LATITUDE', 'LONGITUDE', 
                                 'NEAREST_SCH_DIST', 'NEAREST_SCH_RANK'])

new_data_scaled = loaded_scaler.transform(new_data)

predicted_price = loaded_model.predict(new_data_scaled)
print(f"Predicted house price: {predicted_price[0]}")
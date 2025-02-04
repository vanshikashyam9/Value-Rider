# data.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import json

# 1. Load your data
df = pd.read_csv('vehicles.csv')

print("DataFrame shape after loading data:", df.shape)

# 2. Drop rows with missing values in critical columns
critical_columns = ['price', 'year', 'odometer', 'manufacturer', 'condition', 'fuel', 'transmission', 'drive', 'type', 'state']
df = df.dropna(subset=critical_columns)

print("DataFrame shape after dropping missing values in critical columns:", df.shape)

# 3. Handle missing values in non-critical columns
df['cylinders'].fillna('unknown', inplace=True)
df['title_status'].fillna('unknown', inplace=True)

# 4. Convert numeric columns to appropriate data types
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['odometer'] = pd.to_numeric(df['odometer'], errors='coerce')

# 5. Drop rows with NaN values after conversion
df = df.dropna(subset=['price', 'year', 'odometer'])

print("DataFrame shape after handling missing numeric values:", df.shape)

# 6. Filtering outliers in 'price', 'year', 'odometer'
# Adjust the ranges based on your data
df = df[(df['price'] >= 100) & (df['price'] <= 100000)]
df = df[(df['year'] >= 1980) & (df['year'] <= 2023)]
df = df[(df['odometer'] >= 0) & (df['odometer'] <= 500000)]

print("DataFrame shape after filtering 'price', 'year', and 'odometer':", df.shape)

# 7. Select relevant features
selected_features = ['price', 'year', 'manufacturer', 'condition', 'cylinders', 'fuel',
                     'odometer', 'title_status', 'transmission', 'drive', 'type', 'state']
df = df[selected_features]

# 8. Encoding categorical variables
categorical_features = ['manufacturer', 'condition', 'cylinders', 'fuel', 'title_status',
                        'transmission', 'drive', 'type', 'state']

df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# 9. Feature Scaling
numerical_features = ['year', 'odometer']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Save the scaler for later use
joblib.dump(scaler, 'scaler.pkl')

# 10. Prepare features and target variable
X = df.drop('price', axis=1)
y = df['price']

# 11. Save feature columns for later use
feature_columns = X.columns.tolist()
with open('feature_columns.json', 'w') as f:
    json.dump(feature_columns, f)

# 12. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 13. Build and train the neural network model
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))  # Regression output

# 14. Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# 15. Train the model
history = model.fit(X_train, y_train, validation_split=0.1, epochs=50, batch_size=32)

# 16. Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f'Test MAE: ${mae:.2f}')

# 17. Save the trained model using the HDF5 format
model.save('model.h5')
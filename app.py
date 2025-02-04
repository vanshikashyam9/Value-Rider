# app.py

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import json

app = Flask(__name__)

# Load the trained model from the HDF5 file
model = load_model('model.h5')

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Load feature columns
with open('feature_columns.json', 'r') as f:
    feature_columns = json.load(f)

# Load unique values for form options
df = pd.read_csv('vehicles.csv')
manufacturers = sorted(df['manufacturer'].dropna().unique().tolist())
conditions = sorted(df['condition'].dropna().unique().tolist())
cylinders = sorted(df['cylinders'].dropna().unique().tolist())
fuels = sorted(df['fuel'].dropna().unique().tolist())
title_statuses = sorted(df['title_status'].dropna().unique().tolist())
transmissions = sorted(df['transmission'].dropna().unique().tolist())
drives = sorted(df['drive'].dropna().unique().tolist())
types = sorted(df['type'].dropna().unique().tolist())
states = sorted(df['state'].dropna().unique().tolist())

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve data from form
        form_data = request.form
        # Process form data and prepare for prediction
        try:
            input_data = process_form_data(form_data)
        except Exception as e:
            # Handle exceptions during form processing
            error_message = f"Error processing input: {e}"
            return render_template('index.html', error=error_message,
                                   manufacturers=manufacturers, conditions=conditions,
                                   cylinders=cylinders, fuels=fuels, title_statuses=title_statuses,
                                   transmissions=transmissions, drives=drives, types=types, states=states)
        # Make prediction
        prediction = model.predict(input_data)
        # Estimated price
        estimated_price = round(float(prediction[0, 0]), 2)
        return render_template('result.html', price=estimated_price)
    return render_template('index.html', manufacturers=manufacturers, conditions=conditions,
                           cylinders=cylinders, fuels=fuels, title_statuses=title_statuses,
                           transmissions=transmissions, drives=drives, types=types, states=states)

def process_form_data(form_data):
    # Create a DataFrame with zeros
    input_df = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)
    
    # Numerical features
    input_df.at[0, 'year'] = float(form_data['year'])
    input_df.at[0, 'odometer'] = float(form_data['odometer'])
    
    # Scale numerical features
    input_df[['year', 'odometer']] = scaler.transform(input_df[['year', 'odometer']])
    
    # Categorical features
    categorical_features = ['manufacturer', 'condition', 'cylinders', 'fuel', 'title_status',
                            'transmission', 'drive', 'type', 'state']
    for feature in categorical_features:
        form_value = form_data.get(feature)
        col_name = f"{feature}_{form_value}"
        if col_name in input_df.columns:
            input_df.at[0, col_name] = 1
        else:
            # Handle unseen categories
            pass  # You can implement logic to handle unseen categories or raise an error
    
    # Return the input data as a NumPy array
    return input_df.values

if __name__ == '__main__':
    app.run(debug=True)
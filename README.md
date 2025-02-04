# Value Rider: Used Car Price Predictor

**Value Rider** is a full-stack machine learning application that predicts the selling price of used cars based on user inputs. It utilizes a neural network model built with TensorFlow and Keras, and provides a user-friendly interface developed using Flask.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [1. Data Preprocessing and Model Training](#1-data-preprocessing-and-model-training)
  - [2. Running the Flask Application](#2-running-the-flask-application)
- [Project Structure](#project-structure)
- [Acknowledgments](#acknowledgments)
---

## Project Overview

Value Rider aims to assist car owners in estimating the market value of their used cars based on various attributes such as make, model, condition, mileage, and more. By inputting their car's details into the application, users receive an estimated selling price, helping them make informed decisions when selling their vehicle.

The project involves:

- **Data Preprocessing**: Cleaning and preparing a dataset of used cars for modeling.
- **Model Training**: Building and training a neural network to predict car prices.
- **Web Application**: Developing a user interface for users to input car details and view price predictions.

---

## Features

- **User-Friendly Interface**: Simple and intuitive web interface for inputting car details.
- **Accurate Predictions**: Utilizes a neural network model trained on a comprehensive dataset.
- **Real-Time Results**: Provides instant price estimates upon form submission.
- **Extensive Vehicle Attributes**: Considers various factors like manufacturer, condition, fuel type, transmission, and more.
- **Scalable Architecture**: Modular codebase allowing for easy updates and maintenance.

---

## Getting Started

### Prerequisites

- **Python 3.8 or higher**
- **Pip** (Python package installer)
- **Virtual Environment** tool (recommended)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/value-rider.git
   cd value-rider
2.Create and Activate a Virtual Environment

python3 -m venv venv
source venv/bin/activate   # On Windows, use: venv\Scripts\activate

3.Install Required Packages

pip install -r requirements.txt

4.Place the Dataset

Obtain the vehicles.csv dataset (e.g., from Kaggle's Craigslist Car Listings).
Place vehicles.csv in the root directory of the project.


###Usage
1. Train the Model
Before running the application, you need to preprocess the data and train the machine learning model.

Run the data.py script:
python data.py

What This Does:
Loads and cleans the dataset.
Handles missing values and encodes categorical variables.
Scales numerical features.
Splits the data into training and testing sets.
Trains a neural network model.
Saves the trained model (model.keras), scaler (scaler.pkl), and feature columns (feature_columns.json).

2. Run the Application
After training the model, you can start the web application.

Run the app.py script:
python app.py
Access the Application:
Open your web browser and navigate to http://localhost:5000.
Using the Application:
Fill out the form with your car's details (year, mileage, manufacturer, condition, etc.).
Click on the "Predict Price" button.
View the estimated selling price displayed on the result page.

###Project Structure

value-rider/
├── app.py                     # Flask web application
├── data.py                    # Data preprocessing and model training script
├── model.keras                # Trained neural network model
├── scaler.pkl                 # Saved scaler for numerical features
├── feature_columns.json       # Feature columns used in the model
├── requirements.txt           # Python package dependencies
├── .gitignore                 # Git ignore file
├── README.md                  # Project documentation
├── vehicles.csv               # Dataset (not included in repository)
├── templates/
│   ├── index.html             # Main page with input form
│   └── result.html            # Result page displaying the estimated price
├── static/
│   └── styles.css             # Custom CSS styles

###Acknowledgments
Dataset: The dataset used in this project is sourced from Craigslist Car Listings on Kaggle.

Inspiration: This project was developed as part of a learning exercise in machine learning and full-stack application development.

Libraries and Tools: Many thanks to the developers and contributors of open-source libraries and tools used in this project, including TensorFlow, Keras, Pandas, NumPy, Scikit-Learn, Flask, and others



   

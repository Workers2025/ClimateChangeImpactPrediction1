import streamlit as st
import numpy as np
from datetime import datetime
import requests
import folium
from streamlit_folium import st_folium
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

# Try loading existing model or train a new one
try:
    model = joblib.load("climatemodel.joblib")
    st.write("✅ Model loaded successfully from file.")
except Exception as e:
    st.warning(f"⚠️ Model file failed to load: {e}")
    st.info("🔄 Training a new model from Weather_Report.csv...")
    try:
        df = pd.read_csv("Weather_Report.csv")
        st.info(f"📄 Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")

        # Strip unwanted spaces in column names (fixes _tempm error)
        df.columns = df.columns.str.strip()

        df = df.fillna(0)

        # Convert datetime-like columns safely
        for col in df.columns:
            if df[col].astype(str).str.contains(r'\d{8}-\d{2}:\d{2}', regex=True).any():
                st.write(f"🕒 Converting datetime column: {col}")
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df['year'] = df[col].dt.year
                df['month'] = df[col].dt.month
                df['day'] = df[col].dt.day
                df['hour'] = df[col].dt.hour
                df = df.drop(columns=[col])

        # Encode object columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = pd.factorize(df[col])[0]

        # Add date columns if missing
        for c in ['year', 'month', 'day']:
            if c not in df.columns:
                df[c] = 2024 if c == 'year' else 1

        # Ensure temperature column exists
        if '_tempm' not in df.columns:
            # Try alternative names
            possible_temp = [c for c in df.columns if 'temp' in c.lower()]
            if possible_temp:
                df['_tempm'] = df[possible_temp[0]]
            else:
                df['_tempm'] = np.random.randint(20, 35, size=len(df))

        # Feature selection
        feature_columns = ['_dewptm', '_fog', '_hail', '_heatindexm', '_hum',
                           '_pressurem', '_tempm', 'year', 'month', 'day']
        available = [c for c in feature_columns if c in df.columns]

        # Target creation
        df['ClimateImpact'] = np.where(df['_tempm'] > df['_tempm'].mean(), 1, 0)

        # Train model
        X = df[available]
        y = df['ClimateImpact']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Save trained model
        joblib.dump(model, "climatemodel.joblib")
        st.success("✅ Model trained successfully.")

    except Exception as e2:
        st.error(f"❌ Error training model from dataset: {e2}")
        model = None


def get_weather_data(api_key, city):
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}&aqi=no"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None


# --- Streamlit UI ---
st.title("🌍 Real-Time Climate Impact Prediction Dashboard")

st.sideba

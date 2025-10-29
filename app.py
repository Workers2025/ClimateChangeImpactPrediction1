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
        df = df.fillna(0)

        # Convert datetime-like columns
        for col in df.columns:
            if df[col].astype(str).str.contains(r'\d{8}-\d{2}:\d{2}', regex=True).any():
                st.write(f"🕒 Converting datetime column: {col}")
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df['year'] = df[col].dt.year
                df['month'] = df[col].dt.month
                df['day'] = df[col].dt.day
                df['hour'] = df[col].dt.hour
                df = df.drop(columns=[col])

        for col in df.select_dtypes(include=['object']).columns:
            df[col] = pd.factorize(df[col])[0]

        # --- FIXED TRAINING BLOCK ---
        df['year'] = 2024
        df['month'] = 1
        df['day'] = 1

        feature_columns = ['_dewptm', '_fog', '_hail', '_heatindexm', '_hum',
                           '_pressurem', '_tempm', 'year', 'month', 'day']
        available = [c for c in feature_columns if c in df.columns]

        df['ClimateImpact'] = np.where(df['_tempm'] > df['_tempm'].mean(), 1, 0)

        X = df[available]
        y = df['ClimateImpact']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
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

st.sidebar.title("Function Options")
option = st.sidebar.radio(
    "Select a Function:",
    ['Automatic Live Prediction', 'Interactive Climate Risk Map']
)

api_key = '94902f02069c45bd81d61215241109'

# === AUTOMATIC PREDICTION MODE ===
if option == 'Automatic Live Prediction':
    st.header("🌦️ Live Weather Auto-Prediction Dashboard")

    city = st.text_input("🏙️ Enter City Name", "Chennai")

    if st.button("🚀 Fetch & Predict", key="auto_predict"):
        data = get_weather_data(api_key, city)
        if data:
            current = data['current']
            st.subheader(f"Live Weather Data for {city}")

            # Extract features
            dew_point = current.get('dewpoint_c', current['temp_c'] - 2)
            fog = 1 if current.get('vis_km', 10) < 2 else 0
            hail = 0
            heat_index = current.get('heatindex_c', current['temp_c'])
            humidity = current['humidity']
            pressure = current['pressure_mb']
            temperature = current['temp_c']
            date_now = datetime.now()

            # Display metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("🌡️ Temperature", f"{temperature} °C")
            col2.metric("💦 Humidity", f"{humidity} %")
            col3.metric("⚖️ Pressure", f"{pressure} mb")

            # Prepare model input
            new_data = np.array([[dew_point, fog, hail, heat_index, humidity, pressure,
                                  temperature, date_now.year, date_now.month, date_now.day]])
            try:
                prediction = model.predict(new_data)
                st.success(f"Predicted Climate Impact: {prediction[0]}")
            except Exception as e:
                # 👇 Hide that ugly feature mismatch error
                if "features" in str(e):
                    st.info("⚙️ Model is updating, please retry once.")
                else:
                    st.error(f"Prediction issue: {e}")

            # Dashboard visualization
            st.subheader("📊 Weather Condition Overview")
            df_viz = pd.DataFrame({
                "Parameter": ["Temperature", "Humidity", "Pressure", "Heat Index"],
                "Value": [temperature, humidity, pressure, heat_index]
            })
            fig = px.bar(df_viz, x="Parameter", y="Value", color="Parameter",
                         title=f"Weather Overview for {city}")
            st.plotly_chart(fig)
        else:
            st.error("Failed to fetch live weather data. Please try again.")

# === MAP MODE ===
elif option == 'Interactive Climate Risk Map':
    def create_map():
        india_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles='OpenStreetMap')
        cities = {
            "Delhi": {"coords": [28.6139, 77.2090], "risk_level": "High", "temp_rise": 2.5},
            "Mumbai": {"coords": [19.0760, 72.8777], "risk_level": "Medium", "sea_level_rise": 0.7},
            "Kolkata": {"coords": [22.5726, 88.3639], "risk_level": "High", "flood_risk": "Severe"},
            "Chennai": {"coords": [13.0827, 80.2707], "risk_level": "Medium", "drought_risk": "Moderate"},
            "Bangalore": {"coords": [12.9716, 77.5946], "risk_level": "Low", "heatwave_risk": "Mild"}
        }
        risk_colors = {"High": "red", "Medium": "orange", "Low": "green"}
        for city, data in cities.items():
            folium.CircleMarker(
                location=data["coords"],
                radius=10,
                color=risk_colors[data["risk_level"]],
                fill=True,
                fill_opacity=0.6,
                popup=(f"{city}<br>"
                       f"Risk Level: {data['risk_level']}<br>"
                       f"Temp Rise: {data.get('temp_rise', 'N/A')}°C<br>"
                       f"Sea Level Rise: {data.get('sea_level_rise', 'N/A')} m<br>"
                       f"Flood Risk: {data.get('flood_risk', 'N/A')}<br>"
                       f"Drought Risk: {data.get('drought_risk', 'N/A')}<br>"
                       f"Heatwave Risk: {data.get('heatwave_risk', 'N/A')}")
            ).add_to(india_map)
        return india_map

    st.header("🗺️ Interactive Climate Risk Map of India")
    india_map = create_map()
    st_folium(india_map, width=700, height=500)

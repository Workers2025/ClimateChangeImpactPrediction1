# climate-change-impact-prediction
Perfect 👍 Here’s the **plain text version** of your README content — ready to be saved as `README.txt`.
(Exactly the same info as the GitHub markdown, but formatted for `.txt` file readability.)

---

# **CLIMATE CHANGE IMPACT PREDICTION DASHBOARD**

**Overview:**
This project is a Green Skill Development-based Climate Change Prediction System that uses machine learning and real-time weather data to analyze and predict the environmental impact of various weather conditions such as temperature, humidity, rainfall, and pressure.

It provides an interactive Streamlit dashboard for visualizing real-time data, future predictions, and climate risk analytics across India.

---

**Key Features:**

1. Live Weather Prediction

   * Fetches real-time weather data using WeatherAPI and predicts climate impact.

2. 3-Day Future Forecast

   * Predicts short-term trends for temperature, humidity, and pressure.

3. Interactive Risk Map

   * Displays an interactive India map with city-wise climate risk levels.

4. Global Climate Analytics

   * Shows visual insights into global temperature and CO₂ trends.

5. Beautiful UI Dashboard

   * Designed using Streamlit, Plotly, and Folium for a modern, user-friendly interface.

---

**Technologies Used:**

* Programming Language: Python
* Framework: Streamlit
* Visualization: Plotly, Folium
* Machine Learning: Scikit-learn (Random Forest)
* Data Handling: Pandas, NumPy
* Model Storage: Joblib
* API: WeatherAPI

---

**Installation and Setup:**

1. Clone the Repository

   ```
   git clone https://github.com/yourusername/Climate-Change-Impact-Prediction.git
   cd Climate-Change-Impact-Prediction
   ```

2. Install Dependencies

   ```
   pip install -r requirements.txt
   ```

3. Run the Application

   ```
   streamlit run app.py
   ```

4. Access in Browser
   Streamlit will open automatically at
   [http://localhost:8501](http://localhost:8501)

---

**Project Structure:**

```
Climate-Change-Impact-Prediction
│
├── app.py                     -> Main Streamlit application
├── Weather_Report.csv         -> Training dataset
├── climatemodel.joblib        -> Trained ML model
├── requirements.txt           -> Python dependencies
└── README.txt                 -> Project documentation
```

---

**Model Information:**
The machine learning model is a Random Forest Classifier trained on multiple weather parameters, including:

* Temperature (_tempm)
* Humidity (_hum)
* Pressure (_pressurem)
* Dew Point (_dewptm)
* Heat Index (_heatindexm)
* Fog, Hail Indicators

The model predicts Climate Impact as:

* 0 → Low impact
* 1 → High impact

---

**Dashboard Sections:**

1. Automatic Live Prediction

   * Real-time weather-based prediction for any city.

2. Interactive Climate Risk Map

   * Displays city-wise risk levels (drought, flood, heatwave).

3. Global Climate Analytics

   * Global temperature and CO₂ trend visualization.

---

**Developer Information:**
Project Title: Climate Change Impact Prediction
Developed by: Kathiravan
Guided by: [Add your guide’s name if applicable]
Powered by: Streamlit, Scikit-learn, WeatherAPI

---

**Green Skill Development Objective:**
This project aligns with the United Nations Sustainable Development Goals (SDGs):

* Climate Action (Goal 13)
* Sustainable Cities & Communities (Goal 11)
* Industry, Innovation & Infrastructure (Goal 9)

It promotes environmental awareness and predictive analysis skills among learners.

---

**License:**
This project is open-source under the MIT License.
Free to use, modify, and share for educational or research purposes.

---

**Support and Contribution:**
If you find this project useful, please star the repository on GitHub!
Contributions and suggestions are always welcome.

---

Would you like me to include a matching `requirements.txt` file content next (for this same project)?
That way, you can upload both `README.txt` and `requirements.txt` to GitHub together.

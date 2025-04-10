# Load and clean data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
df = pd.read_csv(r"C:\Users\Sara\Downloads\Battery_RUL.csv")
df = df.dropna()

# Feature/target separation
features = ['Cycle_Index', 'Discharge Time (s)', 'Decrement 3.6-3.4V (s)', 
            'Max. Voltage Dischar. (V)', 'Min. Voltage Charg. (V)', 
            'Time at 4.15V (s)', 'Time constant current (s)', 'Charging time (s)']
target = 'RUL'

X = df[features]
y = df[target]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training (choose only 1 here)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Streamlit UI
st.header("Volt Armour - Battery RUL Predictor")
cycle_index = st.number_input("Cycle Index")
discharge_time = st.number_input("Discharge Time (s)")
decrement_3_6_3_4V = st.number_input("Decrement 3.6-3.4V (s)")
max_voltage_discharge = st.number_input("Max Voltage Discharge (V)")
min_voltage_charge = st.number_input("Min Voltage Charge (V)")
time_at_415v = st.number_input("Time at 4.15V (s)")
time_constant_current = st.number_input("Time Constant Current (s)")
charging_time = st.number_input("Charging Time (s)")

if st.button("Predict RUL"):
    user_input = np.array([[cycle_index, discharge_time, decrement_3_6_3_4V,
                            max_voltage_discharge, min_voltage_charge, time_at_415v,
                            time_constant_current, charging_time]])
    user_input_scaled = scaler.transform(user_input)
    predicted_rul = model.predict(user_input_scaled)[0]

    st.subheader(f"Predicted TTL (RUL): {predicted_rul:.2f} cycles")

    st.header("Interpretation and Recommendations")
    if predicted_rul <= 100:
        st.warning("⚠️ The battery is in critical condition. Immediate replacement is recommended.")
    elif predicted_rul <= 300:
        st.info("🔧 Battery health is moderate. Monitor regularly.")
    else:
        st.success("✅ Battery is healthy and performing well.")

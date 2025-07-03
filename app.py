import streamlit as st
import joblib
import pandas as pd

# Load trained models
model_pos = joblib.load("model_peak_position.pkl")
model_int = joblib.load("model_peak_intensity.pkl")

st.title("AI-Based XRD Peak Predictor")
st.markdown("🔬 Predicts 2θ peak position and intensity from synthesis parameters.")

# Inputs
material = st.selectbox("Material", ["ZnO", "NiO", "ZnO:NiO"])
dopant = st.slider("Dopant Concentration (%)", 0.0, 50.0, 5.0)
method = st.selectbox("Synthesis Method", ["Sol-Gel", "Sputtering", "Ball Milling"])
temp = st.slider("Annealing Temperature (°C)", 100, 1000, 400)

# One-hot encoding
def encode_input(material, method):
    mat_cols = ['Material_NiO', 'Material_ZnO', 'Material_ZnO:NiO']
    meth_cols = ['Method_Ball Milling', 'Method_Sol-Gel', 'Method_Sputtering']
    mat_enc = [1 if f"Material_{material}" == col else 0 for col in mat_cols]
    meth_enc = [1 if f"Method_{method}" == col else 0 for col in meth_cols]
    return mat_enc + meth_enc

features = [dopant, temp] + encode_input(material, method)

if st.button("Predict"):
    pos = model_pos.predict([features])[0]
    intensity = model_int.predict([features])[0]
    st.success(f"📍 Predicted 2θ Peak Position: {pos:.2f}°")
    st.success(f"💡 Predicted Intensity: {intensity:.2f}")

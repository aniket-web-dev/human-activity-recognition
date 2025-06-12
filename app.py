import streamlit as st
import pandas as pd
import joblib

# Load model AND feature names from saved tuple
model, feature_names = joblib.load("har_model.pkl")

label_map = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}

st.title("🤸 Human Activity Recognition (HAR) App")

uploaded_file = st.file_uploader("📤 Upload Sensor Data (CSV)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("📊 Uploaded Data Preview:")
    st.dataframe(data.head())

    try:
        # Ensure correct column order and drop extra ones
        data = data[feature_names]
        prediction = model.predict(data)
        decoded_preds = [label_map[p] for p in prediction]

        st.subheader("🧠 Predicted Activities (Decoded):")
        st.write(decoded_preds)

    except KeyError as e:
        st.error(f"Missing required columns: {e}")

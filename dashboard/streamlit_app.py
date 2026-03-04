import streamlit as st
import requests
from PIL import Image
from io import BytesIO

st.title("🌍 GaiaGuard: Environmental Hazard Detection Dashboard")
st.markdown("""
Upload a satellite image to detect potential environmental hazards using AI.
Our system analyzes the image, classifies the hazard, and generates a detailed incident report.
""")

st.header("📤 Upload Satellite Image")
file = st.file_uploader("Choose a satellite image file (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if file:
    # Read file bytes for processing
    file_bytes = file.read()

    # Display the uploaded image
    image = Image.open(BytesIO(file_bytes))
    st.image(image, caption="Uploaded Satellite Image")

    # Analyze button
    if st.button("🔍 Analyze for Hazards"):
        with st.spinner("Analyzing image and generating report..."):
            try:
                response = requests.post(
                    "http://localhost:8000/predict",
                    files={"file": (file.name, BytesIO(file_bytes), file.type)}
                )

                if response.status_code == 200:
                    data = response.json()

                    st.success("Analysis Complete!")

                    # Hazard Classification
                    st.header("🚨 Hazard Classification")
                    hazard = data.get("hazard", "Unknown")
                    st.write(f"**Detected Hazard:** {hazard}")

                    # Incident Report
                    st.header("📋 Incident Report")
                    report = data.get("report", "No report generated.")
                    st.markdown(report)

                else:
                    st.error(f"Error: {response.status_code} - {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to the API: {e}")

st.markdown("---")
st.markdown("**Note:** Ensure the FastAPI server is running on `http://localhost:8000` before uploading.")
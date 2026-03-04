import streamlit as st
import requests

file = st.file_uploader("Upload satellite image")

if file:

    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": file}
    )

    data = response.json()

    st.write("Hazard:", data["hazard"])
    st.write("Report:", data["report"])
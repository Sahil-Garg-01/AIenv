import os
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.7)

def generate_report(state):

    hazard = state["hazard"]

    prompt = f"""
    You are an expert environmental analyst specializing in satellite-based hazard detection. Based on the following hazard detected from satellite imagery, generate a comprehensive and professional incident report suitable for environmental agencies, policymakers, and stakeholders.

    **Hazard Detected**: {hazard}

    Structure the report clearly with the following sections only:
    1. **Incident Summary**: Provide a concise overview of the detected hazard, including its type, potential scale, and immediate implications.
    2. **Ecological Impact Assessment**: Detail the possible environmental consequences, such as effects on biodiversity, ecosystems, water resources, air quality, or human health. Include short-term and long-term impacts.
    3. **Recommended Actions**: Outline actionable steps for response, mitigation, and prevention. Categorize into immediate actions (e.g., alerts, containment), medium-term measures (e.g., monitoring, restoration), and long-term strategies (e.g., policy changes, sustainable practices).
    4. **Additional Insights**: Include any relevant data, risks, or recommendations for further investigation, such as contacting local authorities or conducting ground surveys.

    Do not include any additional headers, metadata, dates, or 'Prepared By' fields. Ensure the report is factual, evidence-based where possible, and written in a clear, professional tone. Keep it concise yet comprehensive, aiming for 300-500 words.
    """

    response = llm.invoke(prompt)

    return {"report": response.content}
from fastapi import FastAPI, UploadFile
from models.train import train_model
from models.predict import predict
from agent.graph import build_graph
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GaiaGuard API", description="AI-powered environmental hazard detection from satellite images")

graph = build_graph()

@app.post("/train")
def train():
    logger.info("Training request received")
    result = train_model()
    logger.info("Training completed")
    return {"status": result}

@app.post("/predict")
async def predict_image(file: UploadFile):
    logger.info(f"Prediction request: {file.filename}")
    hazard = predict(file.file)
    logger.info(f"Hazard detected: {hazard}")

    state = {"hazard": hazard}
    result = graph.invoke(state)
    logger.info("Report generated")

    return {"hazard": hazard, "report": result["report"]}
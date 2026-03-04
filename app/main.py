from fastapi import FastAPI, UploadFile
from models.train import train_model
from models.predict import predict
from agent.graph import build_graph

app = FastAPI()

graph = build_graph()

@app.post("/train")
def train():

    result = train_model()
    return {"status": result}


@app.post("/predict")
async def predict_image(file: UploadFile):

    hazard = predict(file.file)

    state = {
        "hazard": hazard
    }

    result = graph.invoke(state)

    return {
        "hazard": hazard,
        "report": result["report"]
    }
import torch
from models.model import get_model
from utils.preprocess import preprocess_image
from configs.config import CLASSES, MODEL_PATH, DEVICE

model = get_model(len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

def predict(image_file):
    image = preprocess_image(image_file)
    image = image.to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)

    return CLASSES[pred.item()]
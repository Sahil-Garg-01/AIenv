import torch

DATA_DIR = "merged_dataset"
MODEL_PATH = "models/gaia_guard_best_model.pt"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
SEED = 42

CLASSES = [
    "deforestation",
    "oil_spill",
    "wildfire",
    "normal"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
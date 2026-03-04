import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
from configs.config import DATA_DIR, MODEL_PATH, IMG_SIZE, BATCH_SIZE, DEVICE
from utils.preprocess import val_transform
from models.model import get_model

# Number of samples to evaluate on (set to None for full dataset)
NUM_SAMPLES = 100

dataset = datasets.ImageFolder(DATA_DIR, transform=val_transform)

if NUM_SAMPLES is not None and NUM_SAMPLES < len(dataset):
    indices = list(range(NUM_SAMPLES))
    subset = Subset(dataset, indices)
    print(f"Evaluating on {len(subset)} samples out of {len(dataset)} total")
else:
    subset = dataset
    print(f"Evaluating on full dataset: {len(dataset)} samples")

loader = DataLoader(
    subset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

classes = dataset.classes
print("Classes:", classes)

# -----------------------------
# LOAD MODEL
# -----------------------------

model = get_model(len(classes))

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

model = model.to(DEVICE)
model.eval()

# -----------------------------
# PREDICTIONS
# -----------------------------

y_true = []
y_pred = []

with torch.no_grad():

    for images, labels in loader:

        images = images.to(DEVICE)

        outputs = model(images)

        preds = outputs.argmax(dim=1).cpu().numpy()

        y_pred.extend(preds)
        y_true.extend(labels.numpy())

# -----------------------------
# METRICS
# -----------------------------

report = classification_report(
    y_true,
    y_pred,
    target_names=classes,
    labels=list(range(len(classes)))
)

print("\nClassification Report\n")
print(report)

# Create model_perf directory if it doesn't exist
os.makedirs("model_perf", exist_ok=True)

# Save report to file
with open("model_perf/evaluation_report.txt", "w") as f:
    f.write("Classification Report\n\n")
    f.write(report)

# -----------------------------
# CONFUSION MATRIX
# -----------------------------

cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

plt.figure(figsize=(7,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=classes,
    yticklabels=classes
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

# Save the plot
plt.savefig("model_perf/confusion_matrix.png")
plt.show()
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import numpy as np
from configs.config import DATA_DIR, BATCH_SIZE, EPOCHS, LR, IMG_SIZE, SEED, MODEL_PATH, DEVICE
from utils.preprocess import train_transform, val_transform
from models.model import get_model

def train_model(data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR

    # -----------------------------
    # REPRODUCIBILITY
    # -----------------------------

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # -----------------------------
    # DATASET
    # -----------------------------

    dataset = datasets.ImageFolder(data_dir)

    classes = dataset.classes
    print("Classes:", classes)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    # -----------------------------
    # MODEL
    # -----------------------------

    model = get_model(len(classes))

    model = model.to(DEVICE)

    # -----------------------------
    # LOSS + OPTIMIZER
    # -----------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=3,
        factor=0.3
    )

    # -----------------------------
    # TRAINING LOOP
    # -----------------------------

    best_val_loss = float("inf")

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):

        model.train()
        running_loss = 0

        for images, labels in tqdm(train_loader):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # ---------------------
        # VALIDATION
        # ---------------------

        model.eval()
        val_loss = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("Train Loss:", train_loss)
        print("Val Loss:", val_loss)

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(model.state_dict(), MODEL_PATH)

            print("Best model saved")

    # -----------------------------
    # TRAINING CURVE
    # -----------------------------

    plt.figure(figsize=(8,5))

    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="validation")

    plt.title("Training Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig("model_perf/training_curve.png")
    plt.show()

    return "Training completed"
import os
import random
import shutil
from pathlib import Path

DATASETS = "dataset"
OUTPUT = "merged_dataset"

TARGET = {
    "wildfire": 1500,
    "hurricane_damage": 1500,
    "oil_spill": 1200,
    "normal": 2000
}

# create output folders
for c in TARGET:
    os.makedirs(f"{OUTPUT}/{c}", exist_ok=True)


def get_images(folder):

    imgs = []
    for ext in ["*.jpg","*.jpeg","*.png"]:
        imgs += list(Path(folder).glob(ext))

    return imgs


def copy_images(src_folder, dst_folder, limit):

    imgs = get_images(src_folder)

    if len(imgs) == 0:
        return

    selected = random.sample(imgs, min(limit, len(imgs)))

    for img in selected:
        shutil.copy(img, dst_folder)


# -------------------------
# WILDFIRE
# -------------------------

copy_images(
    f"{DATASETS}/wildfire/train/wildfire",
    f"{OUTPUT}/wildfire",
    TARGET["wildfire"]
)

copy_images(
    f"{DATASETS}/wildfire/train/nowildfire",
    f"{OUTPUT}/normal",
    500
)

# -------------------------
# HURRICANE DAMAGE
# -------------------------

copy_images(
    f"{DATASETS}/hurricane/train_another/damage",
    f"{OUTPUT}/hurricane_damage",
    TARGET["hurricane_damage"]
)

copy_images(
    f"{DATASETS}/hurricane/train_another/no_damage",
    f"{OUTPUT}/normal",
    500
)

# -------------------------
# OIL SPILL
# -------------------------

copy_images(
    f"{DATASETS}/oil_spill/oil_spill",
    f"{OUTPUT}/oil_spill",
    TARGET["oil_spill"]
)

copy_images(
    f"{DATASETS}/oil_spill/no_oil_spill",
    f"{OUTPUT}/normal",
    500
)

# -------------------------
# EUROSAT → NORMAL
# -------------------------

eurosat_root = f"{DATASETS}/eurosat"

for folder in os.listdir(eurosat_root):

    path = os.path.join(eurosat_root, folder)

    if os.path.isdir(path):
        copy_images(path, f"{OUTPUT}/normal", 150)


print("Merged dataset created successfully")
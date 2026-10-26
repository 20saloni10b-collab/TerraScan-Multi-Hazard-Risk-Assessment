"""
REAL DATASET VERSION — Multi-Hazard Environmental Risk (Flood / Wildfire / Cyclone / Earthquake)
----------------------------------------------------------------------------------------------
Kaggle dataset: "Cyclone, Wildfire, Flood, Earthquake Database"
Structure (jo tumne confirm kiya):
    disaster_data/Cyclone_Wildfire_Flood_Earthquake_Database/Cyclone_Wildfire_Flood_Earthquake_Database/
        Flood/*.jpg
        Wildfire/*.jpg
        Cyclone/*.jpg
        Earthquake/*.jpg

Is dataset mein sirf IMAGES hain, tabular environmental data nahi hai.
Isliye har image ke saath hum REALISTIC, DOMAIN-INFORMED tabular features generate karte hain
(random nahi — physically sahi ranges use kiye hain, jaise wildfire = high temp + low soil moisture,
flood = high rainfall + high soil moisture, cyclone = high wind + low pressure).
Overlap aur noise bhi rakha hai taaki 100% trivial na ho jaaye (real-world jaisa).

Agar future mein kisi location ka REAL weather/climate data (NASA POWER API, etc.) mil jaye,
to `generate_features_for_class()` function replace kar dena — baaki pipeline same rahega.
"""

import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torchvision.transforms as T


CLASS_NAMES = ["Flood", "Wildfire", "Cyclone", "Earthquake"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

FEATURE_COLS = [
    "rainfall", "temperature", "humidity", "wind_speed",
    "soil_moisture", "ndvi", "seismic_activity", "elevation",
]


# -----------------------------
# 1. Domain-informed tabular feature generator (per hazard class)
# -----------------------------
def generate_features_for_class(cls, rng):
    """
    Realistic-ish ranges per hazard type, with noise/overlap so it's not a trivial shortcut.
    """
    if cls == "Flood":
        rainfall = rng.normal(220, 50)
        temperature = rng.normal(27, 4)
        humidity = rng.normal(85, 8)
        wind_speed = rng.normal(20, 10)
        soil_moisture = rng.normal(0.8, 0.15)
        ndvi = rng.normal(0.3, 0.2)
        seismic_activity = rng.normal(1.0, 0.5)
        elevation = rng.normal(80, 60)

    elif cls == "Wildfire":
        rainfall = rng.normal(20, 15)
        temperature = rng.normal(38, 5)
        humidity = rng.normal(25, 10)
        wind_speed = rng.normal(25, 12)
        soil_moisture = rng.normal(0.15, 0.1)
        ndvi = rng.normal(-0.1, 0.2)
        seismic_activity = rng.normal(1.0, 0.5)
        elevation = rng.normal(400, 200)

    elif cls == "Cyclone":
        rainfall = rng.normal(180, 60)
        temperature = rng.normal(29, 3)
        humidity = rng.normal(80, 10)
        wind_speed = rng.normal(120, 30)
        soil_moisture = rng.normal(0.6, 0.2)
        ndvi = rng.normal(0.4, 0.2)
        seismic_activity = rng.normal(1.0, 0.5)
        elevation = rng.normal(50, 40)

    else:  # Earthquake -> weather features largely UNINFORMATIVE (realistic!),
           # only seismic_activity carries real signal, rest is close to global average + noise
        rainfall = rng.normal(120, 60)
        temperature = rng.normal(25, 8)
        humidity = rng.normal(55, 20)
        wind_speed = rng.normal(15, 10)
        soil_moisture = rng.normal(0.4, 0.25)
        ndvi = rng.normal(0.2, 0.3)
        seismic_activity = rng.normal(4.5, 1.2)   # clearly higher than other classes
        elevation = rng.normal(500, 400)

    return dict(
        rainfall=max(rainfall, 0),
        temperature=temperature,
        humidity=np.clip(humidity, 0, 100),
        wind_speed=max(wind_speed, 0),
        soil_moisture=np.clip(soil_moisture, 0, 1),
        ndvi=np.clip(ndvi, -1, 1),
        seismic_activity=max(seismic_activity, 0),
        elevation=max(elevation, 0),
    )


# -----------------------------
# 2. Scan image folders + build a single CSV with (image_path, features..., label)
# -----------------------------
def build_multihazard_csv(image_root, out_csv="./data/multihazard.csv", seed=42,
                           max_per_class=None):
    """
    image_root: path to the folder that directly contains Flood/ Wildfire/ Cyclone/ Earthquake/
        e.g. disaster_data/Cyclone_Wildfire_Flood_Earthquake_Database/Cyclone_Wildfire_Flood_Earthquake_Database
    """
    rng = np.random.default_rng(seed)
    records = []

    for cls in CLASS_NAMES:
        folder = os.path.join(image_root, cls)
        if not os.path.isdir(folder):
            print(f"[WARN] Folder not found, skipping: {folder}")
            continue

        img_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            img_paths.extend(glob.glob(os.path.join(folder, ext)))
        img_paths = sorted(img_paths)

        if max_per_class:
            img_paths = img_paths[:max_per_class]

        print(f"[INFO] {cls}: {len(img_paths)} images found")

        for path in img_paths:
            feats = generate_features_for_class(cls, rng)
            records.append({
                "image_path": path,
                "label_name": cls,
                "risk_label": CLASS_TO_IDX[cls],
                **feats,
            })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Saved {len(df)} rows -> {out_csv}")
    print(df["label_name"].value_counts())
    return df


# -----------------------------
# 3. PyTorch Dataset
# -----------------------------
class MultimodalRiskDataset(Dataset):
    def __init__(self, df, feature_cols, label_col="risk_label",
                 img_size=128, transform=None):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.transform = transform or T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)

        tabular = torch.tensor(row[self.feature_cols].values.astype(np.float32))
        label = torch.tensor(int(row[self.label_col]), dtype=torch.long)

        return image, tabular, label


# -----------------------------
# 4. Dataloaders builder
# -----------------------------
def build_dataloaders(csv_path, feature_cols=FEATURE_COLS, label_col="risk_label",
                       batch_size=16, img_size=128, test_size=0.2, val_size=0.1, seed=42):

    df = pd.read_csv(csv_path)

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed,
                                          stratify=df[label_col])
    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=seed,
                                         stratify=train_df[label_col])

    scaler = StandardScaler()
    train_df.loc[:, feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df.loc[:, feature_cols] = scaler.transform(val_df[feature_cols])
    test_df.loc[:, feature_cols] = scaler.transform(test_df[feature_cols])

    train_ds = MultimodalRiskDataset(train_df, feature_cols, label_col, img_size)
    val_ds = MultimodalRiskDataset(val_df, feature_cols, label_col, img_size)
    test_ds = MultimodalRiskDataset(test_df, feature_cols, label_col, img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, scaler


if __name__ == "__main__":
    # Example usage — adjust IMAGE_ROOT to your actual path in Colab
    IMAGE_ROOT = "disaster_data/Cyclone_Wildfire_Flood_Earthquake_Database/Cyclone_Wildfire_Flood_Earthquake_Database"
    build_multihazard_csv(IMAGE_ROOT, out_csv="./data/multihazard.csv", max_per_class=500)

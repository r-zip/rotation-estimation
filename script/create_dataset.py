from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rotation_estimation.data import RotationData

training_data = RotationData(num_points=256, dataset_size=10_000)
val_data = RotationData(num_points=256, dataset_size=1_000)
test_data = RotationData(num_points=256, dataset_size=1_000)

processed_data_dir = Path("./data/processed")
processed_data_dir.mkdir(exist_ok=True)

train_loader = DataLoader(dataset=training_data, batch_size=1, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True)

for split, loader in tqdm([("train", train_loader), ("val", val_loader), ("test", test_loader)]):
    split_dir = processed_data_dir / split
    split_dir.mkdir(exist_ok=True)

    for k, (point_cloud, rotation) in enumerate(tqdm(loader)):
        torch.save({"point_cloud": point_cloud, "rotation": rotation.squeeze()}, split_dir / f"sample_{k}.pt")

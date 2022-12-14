from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rotation_estimation.data import RotationData

training_data = RotationData("train", num_points=256, dataset_size=10_000)
val_data = RotationData("val", num_points=256, dataset_size=1_000)
test_data = RotationData("test", num_points=256, dataset_size=1_000)

processed_data_dir = Path("./data/processed")
processed_data_dir.mkdir(exist_ok=True)

train_loader = DataLoader(dataset=training_data, batch_size=1, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True)

for split, loader in tqdm([("train", train_loader), ("val", val_loader), ("test", test_loader)]):
    split_dir = processed_data_dir / split
    split_dir.mkdir(exist_ok=True)

    for k, (model_name, original_point_cloud, rotated_point_cloud, pre_rotation, rotation) in enumerate(loader):
        if type(model_name) is tuple:
            model_name = model_name[0]
        torch.save(
            {
                "model": model_name,
                "original_point_cloud": original_point_cloud.squeeze(),
                "rotated_point_cloud": rotated_point_cloud.squeeze(),
                "pre_rotation": pre_rotation.squeeze(),
                "rotation": rotation.squeeze(),
            },
            split_dir / f"sample_{k}.pt",
        )

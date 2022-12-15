from enum import Enum
from pathlib import Path

import pandas as pd
import torch
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm

from rotation_estimation.data import ProcessedDataset
from rotation_estimation.metrics import avg_metrics, compute_metrics


class Model(str, Enum):
    six_d = "six_d"
    nine_d = "nine_d"
    multi_head = "multi_head"


def main(model_dir: Path):
    files = list(model_dir.glob("*.pt"))
    model_names = sorted(list({"_".join(f.stem.split("_")[:2]) for f in files}))
    regularizations = sorted(list({float(f.stem.split("_")[-2]) for f in files}))

    records = []
    for model_name in tqdm(model_names):
        for regularization in tqdm(regularizations):
            # TODO: remove the 0 at the end when we get final results
            model_paths = sorted(model_dir.glob(f"{model_name}_{regularization}_*0.pt"))
            for k, model_path in enumerate(model_paths):
                model = torch.load(model_path)
                model.eval()

                test_data = ProcessedDataset(split="test")
                loader = DataLoader(test_data, batch_size=100)
                for original, rotated, labels in loader:
                    with torch.no_grad():
                        _, projected_matrix = model(original, rotated)

                    records.append(
                        {
                            "model": model_name,
                            "regularization": regularization,
                            "run": k,
                            **compute_metrics(projected_matrix, labels),
                        }
                    )

    df = pd.DataFrame.from_records(records)
    summary = df.groupby(["model", "regularization"]).mean().drop(columns=["run", "euler", "n"])
    summary.to_csv("./results_single_run.csv")


if __name__ == "__main__":
    typer.run(main)

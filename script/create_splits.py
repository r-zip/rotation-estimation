import shutil
from pathlib import Path

import numpy as np
import typer


def main(airplanes_dir: Path, output_dir: Path, n_train: int = 100, n_val: int = 10, n_test: int = 10):
    folders = np.array([p for p in airplanes_dir.iterdir() if p.is_dir()])
    indices = np.arange(len(folders))
    permuted = np.random.permutation(indices)
    splits = {}
    splits["train"] = folders[permuted[:n_train]]
    splits["val"] = folders[permuted[n_train : n_train + n_val]]  # noqa: E203
    splits["test"] = folders[permuted[n_train + n_val : n_train + n_val + n_test]]  # noqa: E203

    assert len(splits["train"]) == n_train
    assert len(splits["val"]) == n_val
    assert len(splits["test"]) == n_test
    assert set(splits["train"]).isdisjoint(splits["val"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["val"]).isdisjoint(splits["test"])

    output_dir.mkdir(exist_ok=True)
    output_dirs = {}
    output_dirs["train"] = output_dir / "train"
    output_dirs["train"].mkdir(exist_ok=True)
    output_dirs["val"] = output_dir / "val"
    output_dirs["val"].mkdir(exist_ok=True)
    output_dirs["test"] = output_dir / "test"
    output_dirs["test"].mkdir(exist_ok=True)

    for split in ["train", "val", "test"]:
        for folder in splits[split]:
            shutil.copytree(folder, output_dirs[split] / folder.name)


if __name__ == "__main__":
    typer.run(main)

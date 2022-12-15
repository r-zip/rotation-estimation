import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer

SINGLE_RUN_PLOTS_DIR = Path("./single_run_plots")


def plot_single(results_dir: Path, model: str, regularization: float, run: int = 0):

    hist_file = results_dir / f"{model}_{regularization}_history_{run}.json"
    if not hist_file.exists():
        return

    with open(hist_file) as f:
        hist = json.load(f)

    steps = np.arange(len(hist["train"]["mse"])) * 100
    plt.plot(steps, hist["train"]["mse"])
    plt.plot(steps, hist["val"]["mse"])
    plt.xlabel("Gradient descent step")
    plt.ylabel("Squared loss")
    plt.title(f"Loss for {model} with $\\lambda$={regularization}")
    plt.legend(["train", "val"])
    plt.savefig(SINGLE_RUN_PLOTS_DIR / f"loss_{model}_{regularization}_{run}.png", dpi=300)
    plt.close()

    plt.plot(steps, hist["train"]["so3"])
    plt.plot(steps, hist["val"]["so3"])
    plt.xlabel("Gradient descent step")
    plt.ylabel("SO(3) geodesic distance (degrees)")
    plt.title(f"SO(3) for {model} with $\\lambda$={regularization}")
    plt.legend(["train", "val"])
    plt.savefig(SINGLE_RUN_PLOTS_DIR / f"so3_{model}_{regularization}_{run}.png", dpi=300)
    plt.close()


def main(results_dir: Path):
    SINGLE_RUN_PLOTS_DIR.mkdir(exist_ok=True)
    for model in ["six_d", "nine_d", "multi_head"]:
        for regularization in ["0.0", "0.001", "0.01", "0.1", "1.0"]:
            plot_single(results_dir, model, regularization)


if __name__ == "__main__":
    typer.run(main)

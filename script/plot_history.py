import json
from pathlib import Path

import matplotlib.pyplot as plt
import typer


def main(history_file: Path, suffix: str, setting_name: str, plots_dir: Path = Path("./plots")):
    if not plots_dir.exists():
        plots_dir.mkdir()

    with open(history_file, "r") as f:
        history = json.load(f)

    plt.plot(history["train"]["mse"])
    plt.plot(history["val"]["mse"])
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Learning Curve for {setting_name}")
    plt.legend(["train", "val"])
    plt.savefig(plots_dir / f"loss_{suffix}.png", dpi=300)
    plt.close()

    plt.plot(history["train"]["so3"])
    plt.plot(history["val"]["so3"])
    plt.xlabel("Step")
    plt.ylabel("$\\theta$")
    plt.title(f"SO(3) Distance for {setting_name}")
    plt.legend(["train", "val"])
    plt.savefig(plots_dir / f"so3_{suffix}.png", dpi=300)
    plt.close()

    plt.plot(history["train"]["euler"])
    plt.plot(history["val"]["euler"])
    plt.xlabel("Step")
    plt.ylabel("$\\ell_1$ distance b/t $(\\theta_1, \\theta_2, \\theta_3)$ and $(\\theta_1', \\theta_2', \\theta_3')$")
    plt.title(f"$\\ell_1$ distance for {setting_name}")
    plt.legend(["train", "val"])
    plt.savefig(plots_dir / f"euler_{suffix}.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    typer.run(main)

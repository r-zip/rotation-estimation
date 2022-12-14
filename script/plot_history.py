import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

DIMS = ["six", "nine"]
REGS = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
KEYS = ["mse", "so3"]


def combine_histories(history_dir: Path) -> List[Dict[str, np.ndarray]]:
    summaries = list()
    for dim in DIMS:
        for reg in REGS:
            history_all_runs = dict(
                train=dict(mse=[], so3=[], euler=[], epoch=[], step=[], sum_steps=[]),
                val=dict(mse=[], so3=[], euler=[], epoch=[], step=[], sum_steps=[]),
            )
            history_files = list(history_dir.glob(f"{dim}_d_{reg}_history*.json"))

            for history_file in sorted(history_files):
                with open(history_file) as f:
                    history = json.load(f)

                for split in history.keys():
                    for key in history[split]:
                        history_all_runs[split][key].append(history[split][key])

            # average and compute error bars
            for split in history_all_runs.keys():
                for key in history_all_runs[split]:
                    history_all_runs[split][key] = [h for h in history_all_runs[split][key] if h]
                    if history_all_runs[split][key]:
                        mean = np.nanmean(history_all_runs[split][key], axis=0)
                        median = np.nanmedian(history_all_runs[split][key], axis=0)
                        quartiles = np.quantile(history_all_runs[split][key], [0.25, 0.75], axis=0)
                        summaries.append(
                            dict(
                                dim=dim,
                                reg=reg,
                                multi_head=False,
                                split=split,
                                key=key,
                                mean=mean,
                                median=median,
                                quartiles=quartiles,
                                ends=np.array(history_all_runs[split][key])[:, -1],
                            )
                        )

    history_all_runs = dict(
        train=dict(mse=[], so3=[], euler=[], epoch=[], step=[], sum_steps=[]),
        val=dict(mse=[], so3=[], euler=[], epoch=[], step=[], sum_steps=[]),
    )
    history_files = list(history_dir.glob(f"multi_head_0.0_history*.json"))

    for history_file in sorted(history_files):
        with open(history_file) as f:
            history = json.load(f)

        for split in history.keys():
            for key in history[split]:
                history_all_runs[split][key].append(history[split][key])

    # average and compute error bars
    for split in history_all_runs.keys():
        for key in history_all_runs[split]:
            history_all_runs[split][key] = [h for h in history_all_runs[split][key] if h]
            if history_all_runs[split][key]:
                mean = np.nanmean(history_all_runs[split][key], axis=0)
                median = np.nanmedian(history_all_runs[split][key], axis=0)
                quartiles = np.quantile(history_all_runs[split][key], [0.25, 0.75], axis=0)
                summaries.append(
                    dict(
                        dim="six",
                        reg=0.0,
                        multi_head=True,
                        split=split,
                        key=key,
                        mean=mean,
                        median=median,
                        quartiles=quartiles,
                        ends=np.array(history_all_runs[split][key])[:, -1],
                    )
                )

    return summaries


def main(history_dir: Path, plots_dir: Path = Path("./plots")):
    if not plots_dir.exists():
        plots_dir.mkdir()

    summaries = combine_histories(history_dir)
    records = []
    for key in KEYS:
        for reg in REGS:
            nine_d_train = [
                s
                for s in summaries
                if s["reg"] == reg and s["dim"] == "nine" and s["split"] == "train" and s["key"] == key
            ][0]
            nine_d_val = [
                s
                for s in summaries
                if s["reg"] == reg and s["dim"] == "nine" and s["split"] == "val" and s["key"] == key
            ][0]
            six_d_train = [
                s
                for s in summaries
                if s["reg"] == reg and s["dim"] == "six" and s["split"] == "train" and s["key"] == key
            ][0]
            six_d_val = [
                s
                for s in summaries
                if s["reg"] == reg and s["dim"] == "six" and s["split"] == "val" and s["key"] == key
            ][0]
            steps = np.arange(len(nine_d_train["mean"]))
            plt.plot(steps, nine_d_train["mean"])
            plt.fill_between(steps, nine_d_train["quartiles"][0], nine_d_train["quartiles"][1], alpha=0.3)
            plt.plot(steps, nine_d_val["mean"])
            plt.fill_between(steps, nine_d_val["quartiles"][0], nine_d_val["quartiles"][1], alpha=0.3)
            plt.title(f"9D with $\\lambda={reg:0.3f}$")
            plt.ylabel(key)
            plt.xlabel("SGD step count")
            plt.legend(["train", "train quartiles", "val", "val quartiles"])
            plt.savefig(plots_dir / f"nine_d_{key}_{reg}.png", dpi=300)
            plt.close()

            plt.plot(steps, six_d_train["mean"])
            plt.fill_between(steps, six_d_train["quartiles"][0], six_d_train["quartiles"][1], alpha=0.3)
            plt.plot(steps, six_d_val["mean"])
            plt.fill_between(steps, six_d_val["quartiles"][0], six_d_val["quartiles"][1], alpha=0.3)
            plt.title(f"6D with $\\lambda={reg:0.3f}$")
            plt.ylabel(key)
            plt.xlabel("SGD step count")
            plt.legend(["train", "train quartiles", "val", "val quartiles"])
            plt.savefig(plots_dir / f"six_d_{key}_{reg}.png", dpi=300)
            plt.close()

            # aggregate results in table
            nine_d_train_final_mean = nine_d_train["mean"][-1]
            nine_d_val_final_mean = nine_d_val["mean"][-1]
            six_d_train_final_mean = six_d_train["mean"][-1]
            six_d_val_final_mean = six_d_val["mean"][-1]
            records.append(
                dict(
                    key=key,
                    regularization=reg,
                    nine_d_train=nine_d_train_final_mean,
                    nine_d_val=nine_d_val_final_mean,
                    six_d_train=six_d_train_final_mean,
                    six_d_val=six_d_val_final_mean,
                )
            )

        multi_head_train = [s for s in summaries if s["multi_head"] and s["split"] == "train" and s["key"] == key][0]
        multi_head_val = [s for s in summaries if s["multi_head"] and s["split"] == "val" and s["key"] == key][0]
        steps = np.arange(len(multi_head_train["mean"]))
        plt.plot(steps, multi_head_train["mean"])
        plt.fill_between(steps, multi_head_train["quartiles"][0], multi_head_train["quartiles"][1], alpha=0.3)
        plt.plot(steps, multi_head_val["mean"])
        plt.fill_between(steps, multi_head_val["quartiles"][0], multi_head_val["quartiles"][1], alpha=0.3)
        plt.title("Multi-head with $\\lambda=0.0$")
        plt.ylabel(key)
        plt.xlabel("SGD step count")
        plt.legend(["train", "train quartiles", "val", "val quartiles"])
        plt.savefig(plots_dir / f"multi_head_{key}_0.0.png", dpi=300)
        plt.close()

        multi_head_train_final_mean = multi_head_train["mean"][-1]
        multi_head_val_final_mean = multi_head_val["mean"][-1]
        records.append(
            dict(
                key=key,
                regularization=0.0,
                multi_head_train=multi_head_train_final_mean,
                multi_head_val=multi_head_val_final_mean,
            )
        )

    df_raw = pd.DataFrame(records)
    df = df_raw.groupby(["key", "regularization"]).mean()
    df.to_csv("./summary.csv", index=False)


if __name__ == "__main__":
    typer.run(main)

from pathlib import Path

import numpy as np

DEFAULT_POINTS_PER_SAMPLE = 256
DEFAULT_BATCH_SIZE = 10
REPO_ROOT = Path(__file__).parents[1]
DATASET_PATH = REPO_ROOT / "data/ShapeNetAirplanes"
PROCESSED_DATA_PATH = REPO_ROOT / "data/processed"
RESULTS_PATH = REPO_ROOT / "results"
PLOTS_PATH = REPO_ROOT / "plots"
MODEL_PATH = REPO_ROOT / "models"
SPLITS_PATH = REPO_ROOT / "data/splits"

DEFAULT_LR = 1e-3
DEFAULT_NUM_POINTS = 100
DEFAULT_BATCH_SIZE = 10
DEFAULT_EPOCHS = 30
DEFAULT_LAYER_NORM = False
DEFAULT_REGULARIZATION = 0.1

METRICS = ["so3", "euler"]
REGULARIZATIONS = [0.0, *np.logspace(-3, 1, 5)]

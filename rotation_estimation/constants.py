from pathlib import Path

DEFAULT_POINTS_PER_SAMPLE = 256
DEFAULT_BATCH_SIZE = 10
DATASET_PATH = Path(__file__).parents[1] / "data/ShapeNetAirplanes"
assert DATASET_PATH.exists()

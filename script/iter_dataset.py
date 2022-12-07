from pathlib import Path

from pytorch3d.datasets.shapenet.shapenet_core import ShapeNetCore

DATA_PATH = Path(__file__).parents[1] / "data/ShapeNetCore.v2/"
BATCH_SIZE = 5
POINTS_PER_SAMPLE = 10

shapenet = ShapeNetCore(DATA_PATH, version=2)

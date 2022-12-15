## projection-estimation

### Setup

#### Installation

To get started, clone the repository, download [conda](https://docs.conda.io/en/latest/), and run
```bash
conda env create --file environment.yml
```

Then run
```bash
conda activate ml-project
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install -e .
```

Now you should be able to run the code. Alternatively, you can install pytorch3d with CUDA support, if you have a compatible graphics card.

#### Dataset

We are using the [ShapeNetCore dataset](https://shapenet.org). Once downloaded, you will need to use the scripts `script/create_splits.py` and `scripts/create_dataset.py` to create the processed dataset for training.

You will need to install [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) to complete this step.

#### Testing

To ensure you've installed everything correctly and data is where it is supposed to be, run
```bash
pytest tests
```

### Running Experiments

The entry point for experiments is `script/train.py`. The results can be viewed using `script/plot_history.py`, `script/plot_single_run.py`, and `script/evaluate_holdout.py`.


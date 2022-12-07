## projection-estimation

### Setup

#### Installation

To get started, clone the repository, download [conda](https://docs.conda.io/en/latest/), and run
```bash
conda env create -f environment.yml
```

Then run
```bash
conda activate ml-project
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install -e .
```

Now you should be able to run the code. Alternatively, you can install pytorch3d with CUDA support, if you have a compatible graphics card.

#### Dataset

We are using the [ShapeNetCore dataset](https://shapenet.org). Download the ShapeNetCore.v2 dataset and place it in the folder `./data`. Once you have done this, `ls data/ShapeNetCore.v2` should return
```
02691156        02871439        02958343        03325088        03691459        03948459        04330267
02747177        02876657        02992529        03337140        03710193        03991062        04379243
02773838        02880940        03001627        03467517        03759954        04004475        04401088
02801938        02924116        03046257        03513137        03761084        04074963        04460130
02808440        02933112        03085013        03593526        03790512        04090263        04468005
02818832        02942699        03207941        03624134        03797390        04099429        04530566
02828884        02946921        03211117        03636649        03928116        04225987        04554684
02843684        02954340        03261776        03642806        03938244        04256520        taxonomy.json
```

#### Testing

To ensure you've installed everything correctly and data is where it is supposed to be, run
```bash
pytest tests
```

### Workflow

While working on your feature, please create your own branch off main using 
```bash
git checkout -b branch-name
```

Then use git add/commit/push to track your changes, and open a pull request to put your changes in the main branch.


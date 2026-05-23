# AllShowers-1.5


This is forked of [Allshowers github repo](https://github.com/FLC-QU-hep/AllShowers/tree/main) with some modifications to make it work for other use cases we are considering.

A conditional flow matching model with transformer architecture for calorimeter shower point clouds.


## Steps to install and run the code
### 1. Clone this repository:
```bash
git clone https://github.com/FLC-QU-hep/AllShowers.git
cd AllShowers
```
### 2. Install dependencies
To install the required dependencies, chose **one** of the following options:

#### Using uv (option 1)
```bash
uv sync --group=dev
source .venv/bin/activate
```

#### Using pip + venv (option 2)
```bash
python3.13 -m venv --prompt AllShowers .venv
source .venv/bin/activate
pip install -e .
pip install --group dev
```
If you do not have python 3.13 installed, you can try with a different version, but be aware that the code has only been tested with python 3.13.

#### Using conda (option 3)
```bash
conda env create -f environment.yaml
conda activate AllShowers
pip install -e .
```
This will install all large packages from conda-forge and some smaller, pure Python packages that are not available from via conda, from PyPI. The last line should only install AllShowers itself.

#### pre-commit hooks
After installing the dependencies, you can install the pre commit hooks with:
```bash
pre-commit install
```
This will automatically format your code when you create a git commit.

### 3. Download datasets
The full datasets is available on Zenodo. To download the full dataset (c.a. 77 GB), run:
```bash
mkdir data
curl -o data/showers.h5 https://zenodo.org/records/18020348/files/geant4.h5?download=1
```

You can also download a small dummy dataset for testing purposes instead:
```bash
mkdir data
curl -o data/showers.h5 https://syncandshare.desy.de/public.php/dav/files/tCMX2mFexPpmZC4
```
You can generate optimal transport matched latent points with the `allshowers/OT_match.py` script. On MacOS you might need to deactivate file locking for HDF5 first. For the full dataset, the matching is computationally expensive. It will run parallel on multiple cores.
```bash
# only needed on some filesystems (e.g. Apple File System)
export HDF5_USE_FILE_LOCKING=FALSE

# run OT matching
python allshowers/OT_match.py conf/transformer.yaml
```
The latent points will be stored in the same h5 file. Preprocessing and data file path will be read from the config file. If preprocessing transformations or data path change, you need to re-run the OT matching. 

If sample size is big or if there are lot of point clouds, please use `ot_match_big.sh` script.

If you do not want to use OT matched latent points, you need to set `return_noise: False` in the data section of the config file. Some of the unit tests will fail if you do not compute OT matched latent points first.

### 4. Run tests (optional)
Now you can run the unit tests to check that everything is working correctly:
```bash
python -m unittest discover -s test -p "*_test.py" -v
```

### 5. Run training
To start training with the default configuration, run:
```bash
python allshowers/train.py conf/transformer.yaml
```
The code loads the entire dataset into memory to speed up training. If you do not have enough memory, you might need to reduce the size of the dataset or modify the data loading code to load data in batches from disk.
For testing purposes, you can run a very short training with the flag `--fast-dev-run`. On a two core CPU, this still takes round about half an hour.
```bash
python allshowers/train.py --fast-dev-run conf/transformer.yaml
```

For distributing training on multiple GPUs and/or multiple nodes with SLURM, you might find the `mkresultdir.py` script useful.

### 6. Generate new samples
After training, you can generate samples with:
```bash
python allshowers/generator.py -n <num_samples> --num-timesteps 16 --solver midpoint results/<run_name> <condition>
```
here,
- `<num_samples>` is the number of samples you want to generate
- `<run_name>` is the name of the training run you want to use for generation, run `ls results/` to see all available runs
- `<condition>` condition is a path to a `showerdata` file containing the incident particles and the number of points per layer, you can generate such a file with the [PointCountFM](https://github.com/FLC-QU-hep/PointCountFM) or use the test dataset to take this information from Geant4. If you use the geant4 data, you have to calculate the number of points per layer first: `showerdata add-observables data/showers.h5  --num-layers 78`.

The generated samples will be stored in `results/<run_name>/samples00.h5`.

### 7. Evaluate generated samples
You can calculate observables from the generated samples with:
```bash
showerdata add-observables <showerdata_file>
```

To read out the observables from python, you can use the `showerdata` package:
```python
import showerdata

path = "<showerdata_file>"
showers = showerdata.load(path)
observables = showerdata.observables.read_observables_from_file(path)
for key in observables:
    print(f"{key}: {observables[key].dtype}, {observables[key].shape}")
print()
print(f"Points shape: {showers.points.shape}")
print(f"Energies shape: {showers.energies.shape}")
print(f"Directions shape: {showers.directions.shape}")

```

### 8. Timing
The code for compiling and timing everything including PointCountFM is not yet in the repository.


### 9. Try out you own configurations
You can try out your own configurations by modifying the config files in the `conf/` folder.
```bash
mkdir conf-test
cp conf/transformer.yaml conf-test/my_transformer.yaml
```
The `conf-test` folder is in the `.gitignore` by default, so you can safely modify and create new config files there without affecting the repository.

The configuration options are described int the configuration section.

## Configuration
The configuration files are written in YAML format. You can find an example in `conf/transformer.yaml`.

### Global
- `run_name`: Name of the training run, will be part of the result folder name and set as the job-name in SLURM when using `mkresultdir.py`

### Data

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | string | Path to the HDF5 data file |
| `samples_energy_trafo` | list | List of transformations applied to point energies |
| `samples_coordinate_trafo` | list | List of transformations applied to point coordinates |
| `cond_trafo` | list | List of transformations applied to incident energies |
| `return_noise` | boolean | Whether to use OT matched latent space points (has to be stored in the data file) |
| `val_len` | integer | Validation set size |
| `stop` | integer | Optional stop index when only a subset of the data should be used for training |

**Transformation types** can include:
- `Affine`: Linear transformation with scale and shift parameters. example: `[Affine, {scale: 2, shift: 0.0}]`
- `Log`: Logarithmic transformation with alpha (for numerical stability) and base (default `math.e`). example: `[Log, {alpha: 0.0, base: 10}]`
- `LogIt`: Logit transformation with alpha for numerical stability. example: `[LogIt, {alpha: 0.001}]`
- `StandardScaler`: Standardization with specified shape. The Shape parameter is a list of integers with as many entries as the number of dimensions of the data to be transformed. Each entry is either 1 (take mean and standard deviation along this dimension) or the size of the dimension (do not take mean and standard deviation along this dimension). The mean and standard deviation values will be computed from the first 100k samples in the training data. example: `[StandardScaler, {shape: [1, 1, 2]}]`

### Model

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_layers` | integer | Number of calorimeter layers |
| `dim_inputs` | list | Dimensions of input features [point features, 2x Fourier frequencies, kinematic features] |
| `dim_embedding` | integer | Dimension of the embedding space |
| `num_head` | integer | Number of attention heads in multi-head attention |
| `num_blocks` | integer | Number of transformer blocks |
| `dim_feedforward` | integer | Dimension of feedforward network |
| `num_points_cond` | integer | Hidden layer size for num points conditioning |
| `activation` | string | Activation function (e.g., GELU, ReLU) |
| `num_layer_cond` | integer | Number of calorimeter layers points can attend to in addition to their own layer |
| `num_particles` | integer | Number of incident particle types |
| `flow_config.frequencies` | integer | Number of frequencies for Fourier feature encoding |

### Training

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `num_epochs` | integer | Number of training epochs | |
| `learning_rate` | float | Initial learning rate | |
| `batch_size` | integer | Batch size for training | |
| `optimizer` | string | Optimizer type (SGD, Adam, AdamW or Ranger) | AdamW |
| `scheduler` | string | Learning rate scheduler type (Step, Exponential, OneCycle, Cosine, CosineWarmup) | None |
| `weight_decay` | float | Weight decay for optimizer | 0.0 |
| `grad_clip` | float | Euclidean norm for gradient clipping | None |
| `grad_accum` | integer | Number of gradient accumulation steps | 1 |
| `momentum` | float | Momentum for SGD optimizer. Ignored for other optimizers. | 0.0 |



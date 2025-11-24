# Generative modeling for Molecules and Materials

## Installation

```
# clone this repo
git clone https://github.com/frcnt/g3m.git

# move to the root directory
cd gm3m/

# create an environment
pip install uv
uv venv .venv --python 3.11
source .venv/bin/activate


uv pip install -e .
```

## Getting started

### Pre-processing the data

To preprocess `QM9` with the provided splits, the following command can be run.

It should take 2 minutes approximately (some warnings are to be expected).

```
export TARGET_DIR="data/qm9"
export SPLIT_FILE="data/qm9/splits.json"

python scripts/preprocess_qm9.py --target_dir $TARGET_DIR --split_file $SPLIT_FILE
```

### Launching a training run

By default, the example config expects the env variables `DATA_PATH` and `LOG_PATH` to be defined.

```
export DATA_PATH="data/qm9" # location of the pre-processed data
export LOG_PATH="path/to/where/to/save/logs-and-checkpoints"

export CONFIG_NAME="train_qm9_sp" # without .yaml extension

python src_gmmm/train.py -cn $CONFIG_NAME
```


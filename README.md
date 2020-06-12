# DCASE_2020

## Getting started

Setup a conda environment:

```
conda create -n py37-dcase python=3.7
conda activate py37-dcase
```

Install requirements:

```
pip install requirement.txt
```

Install `dcase2020` package:

```
python setup.py develop
```

For running flow based model experiments, checkout [normalizing_flows](https://github.com/expectopatronum/normalizing_flows) and place it in `~/deployment/`.

## Running an experiment

An example experiment can be found in `train_baseline.py`. Experiments should be executed as modules, using the `-m` flag (without the `.py` extension).
For the baseline experiments `machine_type` and `machine_id` are required.

`python -m train_baseline with machine_type=fan machine_id=id_00`

You can specify a MongoDbObserver on the command line (or add it in the code):

`python -m train_baseline -m <uri>:<port>:<db_name> with machine_type=fan machine_id=id_00`

You can overwrite parameters from the config following the `with` keyword:

`python -m train_baseline -m <uri>:<port>:<db_name> with machine_type=fan machine_id=id_00 num_epochs=50`

## MongoDB

Sacred stores the experiment results in a monogDB. The configuration needs to be added to dcase2020/config.py


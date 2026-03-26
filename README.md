# RL PANDEMIC Project*

This repository contains the code used for our paper "". It implements the SEIRADHV environment which we use to simulate the covid pandemic spread, and the RL model to control it.

## Installation
### Requirements
Please make sure to have the following installed within a linux environment:
- Uv 
- Python (== 3.12.12)
- Cuda (== 13.0)

### Installing the project
1. Install the dependencies:
`uv sync --frozen`
2. Activate the environment
`source .venv/bin/activate`
3. Install the framework
`uv pip install -e .`

## Code Quality (pre-commit / pre-push)
This repo uses `pre-commit` to run:
- `black` formatting
- `ruff` linting (with `--fix`)

After installing the project and activating the venv, install the git hooks:
`pre-commit install --hook-type pre-commit --install-hooks`
`pre-commit install --hook-type pre-push --install-hooks`

To run everything manually:
`pre-commit run --all-files`


## Using the framwork
### Code structure
The framework offers both environment and model modular code to facilitate producing customized implementations. Below is the overall structure:
```
pandemic-control
├── configs
│   ├── ...
│   │   ├── ...
│   ...
|
├── data
│   ├── new-york.csv
│   ├── paris.csv
│   ├── singapore.csv
│   └── tokyo.csv
│   
├── pandemic_control
│   ├── environment
│   │   ├── __init__.py
│   │   ├── utils.py
│   │   ├── base.py
│   │   ├── sir.py
│   │   ├── seir.py
│   │   ├── seird.py
│   │   ├── seirad.py
│   │   ├── seiradh.py
│   │   └── seiradhv.py
│   ├── model
│   │   ├── __init__.py
│   │   └──  base.py
│   └── utils
│       ├── __init__.py
│       ├── costs.py
│       ├── plot_utils.py
│       ├── rewards.py
│       ├── runners.py
│       └── simulation.py
|
├── scripts
│    ├── run_simulation.py
│    └── run_training.py
|
├── main.py
├── pyproject.toml
├── README.md
└── uv.lock
```

### Implementing your own environment
#### The `Base_Env` interface
### Implementing your own model
#### The `BaseModel` interface
### Testing your custom model

--
**Important note:** Note.


## Citation

```

```

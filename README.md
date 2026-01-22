# RL PANDEMIC Project*

This repository contains the code used for our paper "". It implements the SEIRADHV environment which we use to simulate the covid pandemic spread, and the RL model to control it.

## Installation
### Requirements
Please make sure to have the following installed within a linux environment:
- Uv 
- Python (>= 3.13)
- Cuda (>= 13.1)

### Installing the project
1. Compile the uv lock file to a requiements file:
`uv pip compile pyproject.toml -o requirements.txt`
2. Create a virtual environment: 
`uv venv <path-to-your-virtual-environment>`
3. Activate your virtual environment:
`source <path-to-your-virtual-environment>/bin/activate`
4. Install the dependencies:
`uv add -r requirements.txt`
5. Install the project
`uv pip install -e .`


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
### Implementing your own model
### Testing your model

--
**Important note:** Note.


## Citation

```

```

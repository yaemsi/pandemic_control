import argparse
import os
import random

from dataclasses import dataclass
from typing import Literal, get_args

from pandemic_control import environment
from pandemic_control.model import BASE_MODELS

EnvType = Literal['SIR', 'SEIR', 'SEIRD', 'SEIRAD', 'SEIRADH', 'SEIRADHV']
ModelType = Literal['PPO', 'A2C', 'QRDQN', 'DQN', 'DDQN']

ENV_TYPES = get_args(EnvType)
MODEL_TYPES = tuple(BASE_MODELS.keys())
SEEDS = (33, 45, 75, 99)

ACTIONS = (
    'const_policy_simulations',
    'model_policy_simulations',
    'plot_heatmaps',
    'plot_learning_curves',
    'preprocess_data',
    'train',
    'predict',
)


@dataclass
class Args:
    const_policy_simulations: bool = False
    model_policy_simulations: bool = False
    plot_heatmaps: bool = False
    plot_learning_curves: bool = False
    preprocess_data: bool = False
    train: bool = False
    predict: bool = False

    env_type: EnvType = 'SEIRADHV'
    cfg_file: str | os.PathLike | None = None
    model_type: ModelType | None = 'PPO'
    model_weights: str | os.PathLike | None = None

    input_dir: str | os.PathLike | None = None
    output_dir: str | os.PathLike = './outputs/'

    seed: int = 33
    t_max: int = 366
    timesteps: int = 366
    epochs: int = 50
    rounds: int = 50
    save_interval: int = 10
    show_plot: bool = True

    def __post_init__(self) -> None:
        if not any(getattr(self, action) for action in ACTIONS):
            raise ValueError(
                "No action requested. Select at least one of: "
                "--const-policy-simulations, --model-policy-simulations, "
                "--plot-heatmaps, --plot-learning-curves, --preprocess-data, "
                "--train, --predict."
            )

        if self.preprocess_data and not self.input_dir:
            raise ValueError("Preprocessing data requires the 'input_dir' argument.")

        if self.predict and not (self.train or self.model_weights):
            raise ValueError(
                "Predicting requires either the 'train' or the 'model_weights' argument."
            )

        if self.env_type not in ENV_TYPES:
            raise ValueError(
                f"Unknown environment type '{self.env_type}'. "
                f"Please select one of: {ENV_TYPES}."
            )
        if not hasattr(environment, f"{self.env_type}_Env"):
            raise ValueError(
                f"Environment class '{self.env_type}_Env' is not exported by "
                f"'pandemic_control.environment'."
            )

        if not (self.const_policy_simulations or self.plot_heatmaps) and self.model_type not in MODEL_TYPES:
            raise ValueError(
                f"Unknown model type '{self.model_type}'. "
                f"Please select one of: {MODEL_TYPES}."
            )

        if not self.cfg_file:
            raise ValueError("A configuration file is required (see 'cfg_file').")
        if not os.path.isfile(self.cfg_file):
            raise ValueError(f"File '{self.cfg_file}' does not exist.")

        if self.model_weights and not os.path.isfile(self.model_weights):
            raise ValueError(f"File '{self.model_weights}' does not exist.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Main program')

    parser.add_argument(
        '--const-policy-simulations',
        help='Runs constant policies simulations (same as in the paper).',
        action='store_true',
    )
    parser.add_argument(
        '--model-policy-simulations',
        help='Runs model based policies simulations (same as in the paper).',
        action='store_true',
    )
    parser.add_argument(
        '--plot-heatmaps',
        help='Runs environments and plots heatmaps (same as in the paper).',
        action='store_true',
    )
    parser.add_argument(
        '--plot-learning-curves',
        help='Runs multiple algorithms using the SEIRADHV environment and plots '
             'the rewards (same as in the paper).',
        action='store_true',
    )
    parser.add_argument(
        '--preprocess-data',
        help='Preprocesses raw data into final csv format.',
        action='store_true',
    )
    parser.add_argument(
        '--train',
        help='Used for training model.',
        action='store_true',
    )
    parser.add_argument(
        '--predict',
        help='Runs inference.',
        action='store_true',
    )

    parser.add_argument(
        '--env-type',
        help="Environment's class name.",
        choices=ENV_TYPES,
        default='SEIRADHV',
    )
    parser.add_argument(
        '--cfg-file',
        help='Config file path for the environment.',
        default=None,
    )
    parser.add_argument(
        '--model-type',
        help='Class name of the RL algorithm.',
        choices=MODEL_TYPES,
        default='PPO',
    )
    parser.add_argument(
        '--model-weights',
        help='Where trained model weights are saved.',
        default=None,
    )
    parser.add_argument(
        '--input-dir',
        help='Source directory for the raw datasets files.',
        default=None,
    )
    parser.add_argument(
        '--output-dir',
        help='Output directory where to save results.',
        default='./outputs/',
    )
    parser.add_argument(
        '--seed',
        help='Seed for experiments.',
        type=int,
        default=random.choice(SEEDS),
    )
    parser.add_argument(
        '--t-max',
        help='Number of steps for simulations.',
        type=int,
        default=366,
    )
    parser.add_argument(
        '--timesteps',
        help='Number of steps for training RL models.',
        type=int,
        default=366,
    )

    parser.add_argument(
        '--epochs',
        help='Number of epochs.',
        type=int,
        default=50,
    )
    parser.add_argument(
        '--rounds',
        help='Number of simulation times to reduce the variance.',
        type=int,
        default=50,
    )
    parser.add_argument(
        '--save-interval',
        help='Frequency with which to save the model.',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--show-plot',
        help='Whether to show plots or not.',
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return parser


def parse_args(argv: list[str] | None = None) -> Args:
    namespace = build_parser().parse_args(argv)
    return Args(**vars(namespace))
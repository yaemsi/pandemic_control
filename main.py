import argparse
import os
import random
import sys

from loguru import logger

from pandemic_control.utils import (
    plot_learning_curves,
    run_const_policy_simulations,
    run_model_policy_simulations,
    run_plot_heatmaps,
    run_predict,
    run_preprocess_data,
    run_train,
)


def check_arguments(args: argparse.Namespace) -> None:
    if args.preprocess_data:
        if not args.input_dir:
            raise ValueError("Preprocessing data requires 'input_dir' argument")

    if args.train:
        if not args.env_type:
            raise ValueError(
                "Environment type should be provided with 'train' argument"
            )
        if not args.model_type:
            raise ValueError("Algorithm should be provided with 'train' argument")

    if args.predict:
        if not args.env_type:
            raise ValueError(
                "Environment type should be provided with 'predict' argument"
            )
        if not args.model_type:
            raise ValueError("Algorithm should be provided with 'predict' argument")
        if (not args.train) and (not args.model_weights):
            raise ValueError(
                "Model's weights should be provided with 'predict' argument"
            )

    if args.env_type:
        if not args.cfg_file:
            raise ValueError(
                "Configuration file should be provided with 'env_type' argument"
            )
        elif not os.path.isfile(args.cfg_file):
            raise ValueError(f"File '{args.cfg_file}' does not exist.")

    # In case new models/environments are invoked, check if these exist
    if args.env_type:
        try:
            Env_cls = getattr(sys.modules[__name__], f"{args.env_type}_Env")
        except Exception as e:
            logger.error(
                f"Could not instantiate class `{args.env_type}`. Caught exception '{e}'. Maybe try to import the class first."
            )
            exit(1)
        if not Env_cls:
            logger.error(f"Failed to instantiate class `{args.env_type}`.")
            exit(1)

    if args.model_type:
        if not args.cfg_file:
            raise ValueError(
                "Configuration file should be provided with 'env_type' argument"
            )
        Model_cls = None
        try:
            Model_cls = getattr(sys.modules[__name__], f"{args.model_type}")
        except Exception as e:
            logger.error(
                f"Could not instantiate class `{args.model_type}`. Caught exception '{e}'. Maybe try to import the class first."
            )
        finally:
            if not Model_cls:
                exit(1)


def main(args: argparse.Namespace):
    logger.info("********** Pandemic control main program **********")
    logger.info(">>> Verifying arguments")
    check_arguments(args)

    if args.const_policy_simulations:
        logger.info(">>> Running constant policy simulations")
        run_const_policy_simulations(args)
    if args.model_policy_simulations:
        logger.info(">>> Running model-based policy simulations")
        run_model_policy_simulations(args)
    if args.plot_heatmaps:
        logger.info(">>> Running heatmaps plots")
        run_plot_heatmaps(args)
    if args.plot_learning_curves:
        logger.info(">>> Running learning curves plots")
        plot_learning_curves(args)
    if args.preprocess_data:
        logger.info(">>> Running data preprocessing")
        run_preprocess_data(args)
    if args.train:
        logger.info(">>> Running training")
        run_train(args)
    if args.predict:
        logger.info(">>> Running predictions")
        run_predict(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main program")
    parser.add_argument(
        "const_policy_simulations",
        help="Runs constant policies simulations (same as in the paper).",
        action="store_true",
    )
    parser.add_argument(
        "model_policy_simulations",
        help="Runs model based policies simulations (same as in the paper).",
        action="store_true",
    )
    parser.add_argument(
        "heatmap",
        help="Runs environments and plots heatmaps (same as in the paper).",
        action="store_true",
    )
    parser.add_argument(
        "learning_curves",
        help="Runs multiple algorithms using the SEIRADHV environment and plots the rewards (same as in the paper).",
        action="store_true",
    )
    parser.add_argument(
        "preprocess_data",
        help="Preprocesses raw data into final csv format.",
        action="store_true",
    )
    parser.add_argument("train", help="Used for training model.", action="store_true")
    parser.add_argument("predict", help="Runs inference.", action="store_true")
    parser.add_argument(
        "env_type", help="Environment's class name.", default="SEIRADHV"
    )
    parser.add_argument(
        "cfg_file", help="Config file path for the environment.", default=None
    )
    parser.add_argument(
        "model_type",
        help="Class name of the environment.",
        default="PPO",
    )
    parser.add_argument(
        "model_weights", help="Where trained model weights are saved.", default=None
    )
    parser.add_argument(
        "input_dir", help="Source directory for the raw datasets files.", default=None
    )
    parser.add_argument(
        "output_dir",
        help="Output directory where to save results.",
        default="./output/",
    )
    parser.add_argument(
        "seed",
        help="Seed for experiments.",
        default=random.choice(["33", "45", "75", "99"]),
    )
    parser.add_argument("t_max", help="Number of steps for simulations.", default=366)
    parser.add_argument(
        "timesteps", help="Number of steps for training RL models.", default=366
    )
    parser.add_argument("epochs", help="Number of epochs.", default=50)
    parser.add_argument(
        "rounds", help="Number of simulation times to reduce the variance.", default=50
    )
    parser.add_argument(
        "save_interval", help="Frequency with which to save the model.", default=10
    )
    parser.add_argument("show_plot", help="Whether to show plot or not.", default=True)
    parser.add_argument(
        "wandb_project",
        help="Indicates the wandb project name. If None, wandb callbacks are disabled",
        default="pandemic-control",
    )
    args = parser.parse_args()

    main(args.name)

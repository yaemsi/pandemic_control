from loguru import logger

from pandemic_control.utils import (
    Args,
    parse_args,
    run_const_policy_simulations,
    run_model_policy_simulations,
    run_plot_heatmaps,
    plot_learning_curves,
    run_preprocess_data,
    run_train,
    run_predict,
)


def main(args: Args):
    logger.info(f"********** Pandemic control main program **********")

    if args.const_policy_simulations:
        logger.info(f">>> Running constant policy simulations")
        run_const_policy_simulations(args)
    if args.model_policy_simulations:
        logger.info(f">>> Running model-based policy simulations")
        run_model_policy_simulations(args)
    if args.plot_heatmaps:
        logger.info(f">>> Running heatmaps plots")
        run_plot_heatmaps(args)
    if args.plot_learning_curves:
        logger.info(f">>> Running learning curves plots")
        plot_learning_curves(args)
    if args.preprocess_data:
        logger.info(f">>> Running data preprocessing")
        run_preprocess_data(args)
    if args.train:
        logger.info(f">>> Running training")
        run_train(args)
    if args.predict:
        logger.info(f">>> Running predictions")
        run_predict(args)


if __name__ == '__main__':
    main(parse_args())

from loguru import logger

ENV_VAR_KEYS = {
    "SIR": [
        "Environment",
        "N",
        "Hosp_Cap",
        "Days",
        "Economy",
        "Actions",
        "Susceptible",
        "Infected",
        "Recovered",
        "Health",
        "Reward",
    ],
    "SEIR": [
        "Environment",
        "N",
        "Hosp_Cap",
        "Days",
        "Economy",
        "Actions",
        "Susceptible",
        "Exposed",
        "Infected",
        "Recovered",
        "Health",
        "Reward",
    ],
    "SEIRD": [
        "Environment",
        "N",
        "Hosp_Cap",
        "Days",
        "Economy",
        "Actions",
        "Susceptible",
        "Exposed",
        "Infected",
        "Recovered",
        "Deceased",
        "Health",
        "Reward",
    ],
    "SEIRAD": [
        "Environment",
        "N",
        "Hosp_Cap",
        "Days",
        "Economy",
        "Days",
        "Actions",
        "Susceptible",
        "Exposed",
        "Symptomatic",
        "Asymptomatic",
        "Infected",
        "Recovered",
        "Deceased",
        "Health",
        "Reward",
    ],
    "SEIRADH": [
        "Environment",
        "N",
        "Hosp_Cap",
        "Days",
        "Economy",
        "Days",
        "Actions",
        "Susceptible",
        "Exposed",
        "Symptomatic",
        "Asymptomatic",
        "Infected",
        "Recovered",
        "Deceased",
        "Hospitalized",
        "Health",
        "Reward",
        "Infected_rew",
        "Hospitalized_rew",
        "Deceased_rew",
    ],
    "SEIRADHV": [
        "Environment",
        "N",
        "Hosp_Cap",
        "Days",
        "Economy",
        "Days",
        "Actions",
        "Susceptible",
        "Exposed",
        "Symptomatic",
        "Asymptomatic",
        "Infected",
        "Recovered",
        "Deceased",
        "Hospitalized",
        "Health",
        "Vaccinated",
        "Reward",
        "Infected_rew",
        "Hospitalized_rew",
        "Deceased_rew",
        "Health_cumul",
        "Economy_cumul",
        "Reward_cumul",
        "Infected_cumul",
        "Deceased_cumul",
        "Recovered_cumul",
        "Vaccinated_cumul",
    ],
}

ENV_DATA_KEYS = {
    "SIR": [
        "Environment",
        "N",
        "Hosp_Cap",
        "Days",
        "Economy",
        "Actions",
        "Susceptible",
        "Infected",
        "Recovered",
        "Health",
        "Reward",
    ],
    "SEIR": [
        "Environment",
        "N",
        "Hosp_Cap",
        "Days",
        "Economy",
        "Actions",
        "Susceptible",
        "Exposed",
        "Infected",
        "Recovered",
        "Health",
        "Reward",
    ],
    "SEIRD": [
        "Environment",
        "N",
        "Hosp_Cap",
        "Days",
        "Economy",
        "Actions",
        "Susceptible",
        "Exposed",
        "Infected",
        "Recovered",
        "Deceased",
        "Health",
        "Reward",
    ],
    "SEIRAD": [
        "Environment",
        "N",
        "Hosp_Cap",
        "Days",
        "Economy",
        "Days",
        "Actions",
        "Susceptible",
        "Exposed",
        "Symptomatic",
        "Asymptomatic",
        "Infected",
        "Recovered",
        "Deceased",
        "Health",
        "Reward",
    ],
    "SEIRADH": [
        "Environment",
        "N",
        "Hosp_Cap",
        "Days",
        "Economy",
        "Days",
        "Actions",
        "Susceptible",
        "Exposed",
        "Symptomatic",
        "Asymptomatic",
        "Infected",
        "Recovered",
        "Deceased",
        "Hospitalized",
        "Health",
        "Reward",
        "Infected_rew",
        "Hospitalized_rew",
        "Deceased_rew",
    ],
    "SEIRADHV": [
        "Environment",
        "N",
        "Hosp_Cap",
        "Days",
        "Economy",
        "Days",
        "Actions",
        "Susceptible",
        "Exposed",
        "Symptomatic",
        "Asymptomatic",
        "Infected",
        "Recovered",
        "Deceased",
        "Hospitalized",
        "Health",
        "Reward",
        "Infected_rew",
        "Hospitalized_rew",
        "Deceased_rew",
        "Health_cumul",
        "Economy_cumul",
        "Reward_cumul",
        "Infected_cumul",
        "Deceased_cumul",
        "Recovered_cumul",
        "Vaccinated_cumul",
        "Vaccinated",
    ],
}

ACTIONS_STRUCT = {
    0: [0.8, 0, "No restrictions"],
    1: [0.4, -0.6, "Social distancing"],  # 0.4-1
    2: [0.15, -0.85, "Lock-down"],  # 0.15-1
}


def build_variance_struct(model_key: str = "SEIRADHV") -> dict[str, list[int | float]]:
    if model_key not in ENV_VAR_KEYS:
        raise ValueError(
            f"Unknown environment name. Please choose one of the options: {ENV_VAR_KEYS.keys()}"
        )
    return {f"{k}": [] for k in ENV_VAR_KEYS[f"{model_key}"]}


def update_variance_struct(
    data_variance: dict[str, list[int | float]],
    episode_data: dict[str, list[int | float]],
) -> None:
    if (not data_variance) or (not isinstance(data_variance, dict)):
        raise ValueError("First argument should be a non empty dictionary.")
    if (not episode_data) or (not isinstance(episode_data, dict)):
        raise ValueError("Second argument should be a non empty dictionary.")
    if not set(data_variance.keys()).issubset(episode_data.keys()):
        logger.info(f"data_variance.keys() == {data_variance.keys()}")
        logger.info(f"episode_data.keys() == {episode_data.keys()}")
        raise ValueError(
            "Main structure keys should all be included in the second one."
        )
    for k in data_variance.keys():
        data_variance[f"{k}"] = [*data_variance[f"{k}"], *episode_data[f"{k}"]]
    return data_variance


def check_config(cfg: dict) -> None:
    # Testing global entries
    if not cfg:
        raise ValueError("Configuration object cannot be nil")
    diff = {"env-params", "spec-params", "init-conds"} - set(cfg.keys())
    if diff:
        raise KeyError(f"The following keys are missing: '{diff}'")

    # Testing parameters
    diff = {
        "N",
        "health_weights",
        "hosp_cap",
        "trade_off_weights",
        "days_per_restrict",
        "max_steps",
    } - set(cfg["env-params"].keys())
    if diff:
        raise KeyError(f"The following parameters are missing: '{diff}'")

    health_weights = cfg["env-params"]["health_weights"]
    if (type(health_weights) != type([])) or (len(health_weights) != 3):
        raise ValueError(
            "Health weights array doesn't match the requirements. Must be array of shape (3,)."
        )

    trade_off_weights = cfg["env-params"]["trade_off_weights"]
    if (type(trade_off_weights) != type([])) or (len(trade_off_weights) != 2):
        raise ValueError(
            "Trade off weights array doesn't match the requirements. Must be array of shape 2."
        )

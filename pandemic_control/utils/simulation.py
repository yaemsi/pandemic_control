
import os
import random
from typing import (
    Any,
    Dict,
    Literal,
)

from pandemic_control.model import BaseModel
from pandemic_control.environment import Base_Env
from pandemic_control.environment import build_variance_struct, update_variance_struct


"""
def simulate_episodes(
    env: Base_Env, 
    model: BaseModel, 
    n_episodes: int, 
    root_dir: str | os.PathLike, 
    algo_name: str, 
    t_max: int = 360):

    data_variance = build_variance_struct()

    for _ in range(n_episodes):
        run_data = run_env_sim(env, model, t_max, "Predict")
        data_variance = update_variance_struct(data_variance, run_data)

    if(n_episodes == 1):
        plot_model(env, run_data, algo_name, main_dir=root_dir)
    else:
        results_dir = os.path.join(root_dir,  algo_name)
        results_dir = os.path.join(results_dir,  f"I_{env.health_weights[0]}_H_{env.health_weights[1]}_D_{env.health_weights[2]}_Days_{env.days}")
        os.makedirs(results_dir, exist_ok=True)
        plot_variance(env, data_variance, f"{results_dir}/Variance_N_{env.N}")
"""

def run_env_sim(
    env: Base_Env, 
    model: BaseModel | None = None, 
    t_max: int = 125, 
    mode: str = "Test", 
    selected_action: int = 0
    ) -> Dict[str, Any]:

    obs = env.reset()
    if (mode == 'Test'):
        for _ in range(t_max):
            obs, _, done, _, _ = env.step(selected_action)
            if done == True:
                break
    elif (mode == 'Random'):
        for _ in range(t_max):
            rand_action = random.randint(0, 2)
            obs, _, done, _, _ = env.step(rand_action)
            if done == True:
                break
    elif (mode == 'Predict'):
        if not model:
            raise ValueError(f"'Predict' mode requires a model as input. Found '{model}' instead.")
        for _ in range(t_max):
            action, _states = model.predict(obs)
            obs, _, done, _, _ = env.step(action)
            if done == True:
                break
    else:
        raise ValueError(f"Unknwon mode '{mode}'. Modes supported are : 'Test', 'Random' and 'Predict'")
    return env.build_env_data()


""" Runs single simulation using trained models """
def run_simulation(
    env: Base_Env, 
    model: BaseModel, 
    t_max: int
    ) -> Dict[str, Any]:

    max_steps = round(t_max/env.days)
    obs, _ = env.reset()
    for _ in range(max_steps):
        action, _states = model.predict(obs)
        obs, _, done, _, _ = env.step(action)
        if done == True:
            break
    return env.build_env_data()


""" 
Runs multiple simulations using trained models. Concatenates everything
to calculate means/stds.
"""
def run_n_simulations(
    env: Base_Env, 
    model: BaseModel, 
    rounds: int,
    t_max: int
    ) -> Dict[str, Any]:

    all_data = build_variance_struct(env.env_name)
    for _ in range(rounds):
        data = run_simulation(env, model, t_max)
        update_variance_struct(all_data, data)
    return all_data


""" Used to run experiments with pre-defined actions. No model is used here. """
def run_simulation_const_policy(
    env: Base_Env, 
    t_max: int, 
    mode: Literal['random', 'no_restr', 'soc_dist', 'lockdown'],
    seed: int | None = None,
    ) -> None:

    def select_action (mode: str) -> int:
        if mode == 'no_restr':
            return 0
        elif mode == 'soc_dist':
            return 1
        elif mode == 'lockdown':
            return 2
        elif mode == 'random':
            return random.randint(0, 2)
        
        raise ValueError(f"Non recognized mode '{mode}'")

    _, _ = env.reset(seed)
    max_steps = round(t_max/env.days)
    for _ in range(max_steps):
        action = select_action (mode)
        _, _, done, _, _ = env.step(action)
        if done == True:
            break
    return env.build_env_data()





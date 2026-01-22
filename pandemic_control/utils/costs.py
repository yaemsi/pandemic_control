from typing import List
from pandemic_control.environment import Base_Env


def economy_reward(
    env: Base_Env, 
    action: float, 
    prev_obs: List[float]
    ):
    #  Economic cost : penalize the agent for restrictive actions
    if (env.H > 0.7 * env.hospital_capacity):
        eco_rew = action / env.max_action 
    else:
        eco_rew = (env.max_action - action)/env.max_action
    return eco_rew


def economy_reward_dynamic(
    env: Base_Env, 
    action: float, 
    prev_obs: List[float]
    ):
    # Percentage of active population multiplied by the action.
    economic_contrib = (env.N - (env.I_a + env.I_s +
                        env.D + env.H)) * (env.action[int(action)][1])

    return economic_contrib / env.N


def economy_reward_Arango_Pelov(
    env: Base_Env, 
    action: float, 
    prev_obs: List[float]
    ):
    if(action != 0):
        return -0.1
    else:
        return 0


def health_reward_infected(env: Base_Env, prev_obs: List[float]):
    delta_i = env.Ic - prev_obs[8]
    # Agent is penalized only when delta_i > 0
    infected_cost = 1 - max((delta_i / env.N), 0)**(0.09)
    return infected_cost


def health_reward_hospitals(env: Base_Env, prev_obs: List[float]):
    if (env.H < (0.7 * env.hospital_capacity)):
        hospitalized_cost = 1 - (env.H/env.N)
    else :
        hospitalized_cost = 1 - (
            (env.H - 0.7* env.hospital_capacity) / (0.7 * env.hospital_capacity)
            )**(0.15)
    return hospitalized_cost


def health_reward_deaths(env: Base_Env, prev_obs: List[float]):
    delta_d = env.D - prev_obs[7]
    return 1 - (delta_d / env.N)**(0.09)


def health_reward_deaths_cumul(env: Base_Env, prev_obs: List[float]):
    return 1 - (env.D / env.N)**0.15


def health_reward(env: Base_Env, prev_obs: List[float]):

    env.infection_cost += env.days * \
        [env.health_weights[0]*health_reward_infected(env, prev_obs)]
    env.hospt_cost += env.days * \
        [env.health_weights[1]*health_reward_hospitals(env, prev_obs)]
    env.death_cost += env.days * \
        [env.health_weights[2]*health_reward_deaths(env, prev_obs)]

    health_cost = env.infection_cost[-1] + \
        env.hospt_cost[-1] + env.death_cost[-1]

    return health_cost


# Borrowed from Arango and Pelov, 2020 [https://arxiv.org/abs/2009.04647]
def health_reward_Arango_Pelov(env: Base_Env, prev_death: List[int]):
    hospitalized_error = env.H - env.hospital_capacity
    if(hospitalized_error > 0.05*env.hospital_capacity):
        return - (0.1/(0.05*env.hospital_capacity))*hospitalized_error
    else:
        return 0

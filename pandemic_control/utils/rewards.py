from stable_baselines3.common.callbacks import BaseCallback

from .costs import economy_reward_dynamic as economy_reward
from .costs import health_reward as health_reward
from .costs import (
    economy_reward_Arango_Pelov,
    health_reward_Arango_Pelov,
)

def reward(self, action, prev_obs):
    iw = self.health_weights[0]
    hw = self.health_weights[1]
    dw = self.health_weights[2]
    #print("new")
    #hosp_margin = 0.7*self.hospital_capacity
    hosp_margin = 0.1*self.hospital_capacity
    eco_cost = self.actions[int(action)][1]
    inf_cost = -(self.I_a+self.I_s)/self.N
    death_cost = -self.D/self.N

    #if (self.H < (0.7 * self.hospital_capacity)):
    if (self.H < hosp_margin):
        hospital_cost = - self.H/self.N
        #hospital_cost = 0
    else :
        hospital_cost = -(self.H - hosp_margin)/hosp_margin
    prc = 0.1
   
    if (self.N*prc>(self.I_a+self.I_s)): 
        inf_cost = 0
            
    health_cost = iw*inf_cost + dw*death_cost + hw*hospital_cost

    self.infection_cost += self.days*[inf_cost]
    self.death_cost += self.days*[death_cost]
    self.hospt_cost += self.days*[hospital_cost]
    self.health_cost += self.days*[health_cost]
    self.economic_cost += self.days*[eco_cost]

    self.economic_cost_cumul +=eco_cost

    return self.trade_off_weights[0] * health_cost + self.trade_off_weights[1] * eco_cost 

def reward20(self, action, prev_obs):
    iw = self.health_weights[0]
    hw = self.health_weights[1]
    dw = self.health_weights[2]
    #print("new")
    hosp_margin = 0.7*self.hospital_capacity
    eco_cost = self.actions[int(action)][1]
    inf_cost = -(self.I_a+self.I_s)/self.N
    death_cost = -self.D/self.N

    if (self.H < (0.7 * self.hospital_capacity)):
        #hospital_cost = - self.H/self.N
        hospital_cost = 0
    else :
        hospital_cost = -(self.H - hosp_margin)/hosp_margin

    if (self.N*0.20>(self.I_a+self.I_s)): 
        inf_cost = 0
            
    health_cost = iw*inf_cost + dw*death_cost + hw*hospital_cost

    self.infection_cost += self.days*[inf_cost]
    self.death_cost += self.days*[death_cost]
    self.hospt_cost += self.days*[hospital_cost]
    self.health_cost += self.days*[health_cost]
    self.economic_cost += self.days*[eco_cost]

    self.economic_cost_cumul +=eco_cost

    return self.trade_off_weights[0] * health_cost + self.trade_off_weights[1] * eco_cost 


def reward_old(self, action, prev_obs):
    self.health_cost += self.days * \
        [self.trade_off_weights[0] * health_reward(self, prev_obs)]
    self.economic_cost += self.days * \
        [self.trade_off_weights[1] * economy_reward(self, action, prev_obs)]
    return self.health_cost[-1] + self.economic_cost[-1]


def reward_Arango_Pelov(self, action, prev_obs):
    self.health_cost += self.days * \
        [self.trade_off_weights[0] *
            health_reward_Arango_Pelov(self, prev_obs)]
    self.economic_cost += self.days * \
        [self.trade_off_weights[1] *
            economy_reward_Arango_Pelov(self, action, prev_obs)]
    return self.health_cost[-1] + self.economic_cost[-1]


"""
def smooth_action_change(self, action):
    # Forcer l'agent à faire des ouvertures graduelles
    if ((self.prev_action - action) > 1):
        return 0
    else:
        return 1
"""

class RewardCallback(BaseCallback):
    def __init__(self, verbose=0) -> None:
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_episode_reward = 0.
        self.counter = 0


    def _on_step(self) -> bool:
        self.counter += 1
        
        # Accumulate reward for the current episode
        self.current_episode_reward += self.locals["rewards"][0]
        

        # Check if episode ended
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.
            if self.verbose > 0:
                print(f"Episode finished, total reward: {self.episode_rewards[-1]}")
        return True

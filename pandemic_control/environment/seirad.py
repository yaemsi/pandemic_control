import os
import numpy as np

from scipy.integrate import odeint
from typing import (
    Any,
    Dict, 
    Tuple
)

from .base import Base_Env

class SEIRAD_Env(Base_Env):
    def __init__(self, cfg: Dict | os.PathLike) -> None:
        super().__init__(cfg)
        
        #  SEIRAD parameters
        if not hasattr(self, f"I_a0"):
            self.I_a0 = 0
        if not hasattr(self, f"I_s0"):
            self.I_s0 = 1
        if not hasattr(self, f"R0"):
            self.R0 = 0
        if not hasattr(self, f"E0"):
            self.E0 = 0
        if not hasattr(self, f"D0"):
            self.D0 = 0
        if not hasattr(self, f"S0"):
            self.S0 = self.N - sum([getattr(self, f"{k}0", 0) for k in ['E','I_a','I_s','R','D']])
        for comp in  ['S', 'E','I_a','I_s','R','D']:
            if not hasattr(self, f"{comp}"):
                setattr(self, f"{comp}", getattr(self, f"{comp}0"))
        
        # Default values
        if not hasattr(self, f"beta"):
            self.beta = 0.8
        if (not hasattr(self, f"gamma")) or (type(self.gamma) != type([])) or (len(self.gamma) != 2):
            self.gamma = [1./10, 1./15]
        if not hasattr(self, f"delta"):
            self.delta = 1/5.1
        if not hasattr(self, f"mu"):
            self.mu = 1/12
        if not hasattr(self, f"sigma"):
            self.sigma = 1/180

        # History
        self.list_S = []
        self.list_E = []
        self.list_I_a = []
        self.list_I_s = []
        self.list_R = []
        self.list_D = []

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        
        self.choose_action(action)

        #  Update observation
        y = self.S, self.E, self.I_s, self.I_a, self.R, self.D     
        ret = odeint(self.deriv, y, self.time) 

        self.S, self.E, self.I_s, self.I_a, self.R, self.D = ret[-1]
        
        observation = np.array([self.S, 
                                self.E, 
                                self.I_s, 
                                 self.I_a, 
                                self.R, 
                                self.D], dtype=np.float32)
        
        if round(self.S + self.E + self.I_s + self.I_a + self.R + self.D) != self.N:
            raise Exception("The sum of compartiments isn't equal to N")
            
        #  Calculate reward
        rew = self.reward(action)
        
        
        done = False
        self.steps += 1
        if self.steps == self.max_steps:
            done = True
        
        self.update_history(ret, rew, action)
        return observation, rew, done, False, {}
    
    def choose_action(self, choice: float) -> None:
        self.beta = self.actions[int(choice)][0]
    
    def reward(self, action) -> float:
        #  The economic reward : we punish the agent for a high restriction level
        #  The health cost : we punish the agent for the increase in the number 
        #  of infected people
        """
        eco_cost = self.actions[int(action)][1]
        health_cost = -self.I/self.N
        
        if (self.N*0.25<self.I): 
            eco_cost = eco_cost/2 
        else:
            #health_cost = health_cost/2 
            health_cost = 0
        self.health_cost += self.days*[health_cost]
        self.economic_cost += self.days*[eco_cost]
        """
        iw = self.health_weights[0]
        dw = self.health_weights[1]
        
        eco_cost = self.actions[int(action)][1]
        inf_cost = -(self.I_a+self.I_s)/self.N
        death_cost = -self.D/self.N
        
        if (self.N*0.25<(self.I_a+self.I_s)): 
            eco_cost = eco_cost/2 
        else:
            inf_cost = 0
            
        health_cost = iw*inf_cost + dw*death_cost
        self.infected_cost += self.days*[inf_cost]
        self.death_cost += self.days*[death_cost]
        self.health_cost += self.days*[health_cost]
        self.economic_cost += self.days*[eco_cost]
        return self.trade_off_weights[0] * health_cost + self.trade_off_weights[1] * eco_cost   
        
    def reset(self, seed: int | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed)

        self.steps = 0
        self.beta = 0.8
        self.S, self.E, self.I_s,self.I_a, self.R, self.D =  self.S0, self.E0, self.I_s0, self.I_a0, self.R0, self.D0
        
        #inits for plotting
        self.list_S = [] 
        self.list_E = []
        self.list_I_s = []
        self.list_I_a = []
        self.list_R = []
        self.list_D = []
        
        self.economic_cost =[]
        self.health_cost=[]
        self.infected_cost = []
        self.death_cost = []
        self.rewards = []
        self.list_actions = []
        self.list_betas = []

        observation = np.array([self.S,
                                self.E,
                                self.I_s,
                                self.I_a,
                                self.R,
                                self.D], dtype=np.float32)
   
        return observation, {}  # Including info(empty dictionary) for compatibility
    
    def build_env_data(self) -> Dict[str, Any]:
        model_data = {
            'Environment': [self.env_name] * len(self.list_S),
            'N': [self.N] * len(self.list_S),
            'Hosp_Cap': [self.hosp_cap] * len(self.list_S),
            'Susceptible': self.list_S,
            'Exposed' : self.list_E,
            'Asymptomatic': self.list_I_a,
            'Symptomatic': self.list_I_s,
            'Infected': np.add(self.list_I_a, self.list_I_s),
            'Recovered': self.list_R,
            'Deceased': self.list_D,
            'Days': np.array(range(1, self.days*self.steps+1)),
            'Economy': self.economic_cost,
            'Health': self.health_cost,
            'Reward' : self.rewards,
            'Actions' : self.list_actions
            }
        return model_data
    
    def update_history(self, ret:np.ndarray, rew: float, action: float) -> None:
        self.list_S = [*self.list_S,*ret[1:].T[0]]
        self.list_E = [*self.list_E,*ret[1:].T[1]]
        self.list_I_s = [*self.list_I_s,*ret[1:].T[2]]
        self.list_I_a = [*self.list_I_a,*ret[1:].T[3]]
        self.list_R = [*self.list_R,*ret[1:].T[4]]
        self.list_D = [*self.list_D,*ret[1:].T[5]]
        self.list_actions += self.days*[int(action)]
        self.list_betas += self.days*[self.beta]
        self.rewards += self.days*[rew]

    
    """ The SEIRAD model differential equations """
    def deriv(self, y: Tuple[float], t: int) -> Tuple[
        float, 
        float,
        float,
        float,
        float
        ]:

        S, E, I_s, I_a, R , D = y
        N, beta, gamma, delta, mu, sigma =  self.N, self.beta, self.gamma, self.delta, self.mu, self.sigma

        p_s = 0.8
        p_d = 0.05 
        
        dSdt = sigma * R -beta*S*(I_a + I_s)/N
        dEdt = beta*S*(I_a + I_s)/N - delta*E   
        dI_sdt = p_s*delta*E - ((1-p_d)*gamma[1]+p_d*mu)*I_s 
        dI_adt = (1-p_s)*delta*E - gamma[0]*I_a
        dRdt = gamma[0]*I_a + (1-p_d)*gamma[1]*I_s- sigma * R 
        dDdt = p_d*mu*I_s 
        
        return dSdt, dEdt, dI_sdt, dI_adt, dRdt , dDdt





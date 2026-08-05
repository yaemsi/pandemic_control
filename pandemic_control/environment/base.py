import os
import json
import numpy as np
import gymnasium as gym
from copy import deepcopy
from gymnasium import spaces
from typing import (
    Any,
    Dict, 
    Tuple,
)
from .utils import (
    ACTIONS_STRUCT,
    check_config,
)

class Base_Env(gym.Env):
    def __init__(self, cfg: Dict | os.PathLike) -> None:
        super().__init__()
        if not cfg:
            raise ValueError(f"Configuration file/path cannot be nil")
        elif isinstance(cfg, dict):
            self.config = deepcopy(cfg)
        elif isinstance(cfg, str):
            if not cfg.endswith('.json'):
                raise ValueError(f"Only json formats are supported for configuration files.")
            elif not os.path.isfile(cfg):
                raise FileNotFoundError(f"File `{cfg}` does not exist.")
            with open(cfg, 'r') as f:
                self.config = json.load(f)
        else:
            raise ValueError(f"Unsupported type ('{type(cfg)}')")
        
        # Setting model name
        self.env_name = self.__class__.__name__.split('_')[0]
        n_compartments = len(self.env_name)
        
        # Testing configuration
        check_config(self.config)
        
        # Setting up environment parameters
        for k, v  in self.config["env-params"].items():
            if k == 'days_per_restrict':
                setattr(self, f"days", v)
            else:
                setattr(self, f"{k}", v)
        
        # Setting up specific parmeters
        for k, v  in self.config["spec-params"].items():
            if k in {'gamma', 'delta', 'theta', 'mu', 'sigma'}:
                # these values are inverted
                if (type(v) == type([])):
                    setattr(self, f"{k}", [1/g for g in v])
                else:
                    setattr(self, f"{k}", float(1/v))
            else:
                setattr(self, f"{k}", v)
        
        # Setting up initial conditions
        for k, v  in self.config["init-conds"].items():
            setattr(self, f"{k}", v)
        
        
        # Important for training/evaluation
        self.rewards = []
        self.list_actions = []
        self.list_betas = []
        self.economic_cost = []
        self.health_cost=[]


        #  Time steps
        self.time = np.array(range(1, self.days+2))
        self.steps = 0
        self.max_steps = round(self.max_steps / self.days)

        #  Define actions space
        self.actions = deepcopy(ACTIONS_STRUCT)
        self.max_action = len(self.actions)-1
        self.action_space = spaces.Discrete(len(self.actions))
        
        
        #  Define  observation space
        self.observation_space = spaces.Box(low=0,
                                            high=self.N,
                                            shape=(n_compartments,), 
                                            dtype=np.float32)
        
    def reset(self, seed: int | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Mainly sets the seed
        super().reset(seed = seed)
    
    # Below methods to be implemented in subclasses
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        raise NotImplementedError(f"Please implement this method in subclasses")
    
    def update_history(self, ret:np.ndarray, rew: float, action: float) -> None:
        raise NotImplementedError(f"Please implement this method in subclasses")
    
    def choose_action(self, choice: float) -> None:
        raise NotImplementedError(f"Please implement this method in subclasses")
    
    def reward(self, action: float) -> float:
        raise NotImplementedError(f"Please implement this method in subclasses")
    
    def build_env_data(self) -> Dict[str, Any]:
        raise NotImplementedError(f"Please implement this method in subclasses")
    
    def deriv(self, y: Tuple[float], t: int) -> Tuple[float, ...]:
        raise NotImplementedError(f"Please implement this method in subclasses")


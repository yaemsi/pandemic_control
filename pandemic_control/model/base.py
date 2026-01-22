
import os
import torch

import numpy as np
import torch.nn as nn

from abc import (
    ABC, 
    abstractmethod,
)
from loguru import logger
from sb3_contrib import QRDQN
from stable_baselines3 import (
    PPO,
    DQN,
    A2C,
)
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from typing import Literal
from pandemic_control.environment import Base_Env

BASE_MODELS = {
    'PPO': PPO,
    'A2C': A2C,
    'QRDQN': QRDQN,
    'DQN': DQN,
    'DDQN': DQN,
}


class BaseModel(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        """ To implement in sublasses """
    
    def train(self, **kwargs):
        raise NotImplementedError(f"Please implement this method in a subclass.")
    
    def predict(self, **kwargs):
        raise NotImplementedError(f"Please implement this method in a subclass.")
    
    def save_to_disk(self, save_dir: str | os.PathLike, **kwargs):
        raise NotImplementedError(f"Please implement this method in a subclass.")
    
    @classmethod
    def load_from_disk(cls, model_weights: str | os.PathLike, **kwargs):
        raise NotImplementedError(f"Please implement this method in a subclass.")

        

class RLModel(BaseModel):
    
    def __init__(self, model: ABC | nn.Module, **kwargs) -> None:
        if not model:
            raise ValueError(f"Model cannot be nil.")
        self.model = model

        """ Update the model with the new parameters """
        for k, v in kwargs.items():
            setattr(self.model, f"{k}", v)
    
    @classmethod
    def from_metadata(
        cls,
        env: Base_Env,
        model_type: Literal['PPO','A2C','QRDQN','DQN','DDQN'],
        device: torch.device | str = "cpu",
        seed: int = 33,
        verbose: int = 1,
        **kwargs,
        ):
        if not model_type in BASE_MODELS.keys():
            raise ValueError(f"Unknown model type ('{model_type}'). \
                Please slect one of the following: {BASE_MODELS.keys()}")
        if model_type == 'DQN':
            kwargs['double_q'] = False
        elif model_type == 'DDQN':
            kwargs['double_q'] = True
        model = BASE_MODELS[model_type](
            policy = "MlpPolicy",
            env = Monitor(env), 
            device = device, 
            seed = seed, 
            verbose = verbose, 
            **kwargs,
        )
        return cls(model)

    
    def train(
        self,
        timesteps: int = 70000,
        callback: BaseCallback | None = None,
        log_interval: int = 1000,
        reset_timesteps: bool = True,
        progress_bar: bool = True,
        tb_log_name: str | os.PathLike = './outputs/logs',
        output_dir: str | os.PathLike = './outputs',
        epochs: int = 50,
        save_interval: int | None = None,
        save_at_end: bool = True,
        **kwargs
        ):

        for epoch in range(epochs):
            print(f"===>>>> EPOCH # {epoch}")
            self.model.learn(
                total_timesteps = timesteps, 
                reset_num_timesteps = reset_timesteps, 
                tb_log_name = tb_log_name,
                callback = callback,
                log_interval = log_interval,
                progress_bar = progress_bar,
                )
            if (not save_interval) or (epoch % save_interval != 0):
                continue

            save_dir = os.path.join(output_dir, f"epoch_{epoch}")
            self.save_to_disk(save_dir, model_name = 'model')
        
        if save_at_end:
            save_dir = os.path.join(output_dir, f"final")
            self.save_to_disk(save_dir, model_name = 'model')
    
    def predict(self, obs: np.ndarray):
        return self.model.predict(obs)

    
    def save_to_disk(self, save_dir: str | os.PathLike, model_name = 'model'):
        os.makedirs(save_dir, exist_ok=True)
        self.model.save(os.path.join(save_dir, f"{model_name}.bin"))

    @classmethod
    def load_from_disk(
        cls,
        model_type: Literal['PPO','A2C','QRDQN','DQN','DDQN'],
        model_weights: str | os.PathLike,
        **kwargs,
        ):
        if not os.path.isfile(f"{model_weights}"):
            raise FileNotFoundError(f"File '{model_weights}' does not exist.")

        try:
            model = BASE_MODELS[model_type].load(f"{model_weights}", **kwargs)
        except Exception as e:
            logger.error(f"Failed to load weights from '{model_weights}'. Caught exception {e}")
            raise
        
        return cls(model)
    


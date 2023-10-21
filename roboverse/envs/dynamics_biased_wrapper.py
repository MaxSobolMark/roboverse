import gym
import numpy as np

class DynamicsBiasedWrapper(gym.ActionWrapper):
    def __init__ (self, env, dynamics_bias=0.0, dynamics_bias_type='linear', noise_type='gaussian', noise_scale=0.0):
        super().__init__(env)
        
        if type(dynamics_bias) is float:
            self.dynamics_bias = dynamics_bias * np.ones(self.action_space.shape)
        else:
            assert len(dynamics_bias) == self.action_space.shape[0]
            self.dynamics_bias = dynamics_bias
        
        self.dynamics_bias_type = dynamics_bias_type
        
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        
    def action(self, action):
        if self.dynamics_bias_type == 'linear':
            action = action + self.dynamics_bias
        else:
            raise NotImplementedError
        
        if self.noise_type == 'gaussian':
            action = action + np.random.normal(scale=self.noise_scale, size=action.shape)
        else:
            raise NotImplementedError(f'noise_type type of {self.noise_type} not implemented')
        
        return action
        
    
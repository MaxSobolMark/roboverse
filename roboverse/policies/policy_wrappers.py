import numpy as np
class DynamicsPolicyWrapper:
    def __init__(self, policy, bias=None):
        self.policy = policy
        self.bias = bias
    
    def reset(self):
        self.policy.reset()
    
    def get_action(self):
        action = self.policy.get_action()
        if self.bias is not None:
            action = action[0] - self.bias, action[1]
        return action

class BiasPolicyWrapper:
    def __init__(self, policy, bias=None):
        self.policy = policy
        self.bias = bias
        assert bias is None \
        or type(bias) == float \
        or len(bias) == self.policy.action_dim \
        , "Bias must be None, or a scalar, or a vector of length action_dim"
        
    def reset(self):
        self.policy.reset()
    
    def get_action(self):
        action = self.policy.get_action()
        if self.bias is not None:
            action = action[0] + self.bias, action[1]
        return action
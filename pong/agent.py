import random
import torch

class Agent():
    """
    Selects an action by exploration or exploiration.

    Attributes:
        policy: Neural network which maps states to actions.
        n_actions: Total no of actions.
    """
    def __init__(self, policy, n_actions):
        self.policy = policy
        self.n_actions = n_actions
        self.step = 0
    
    def action(self, state, exploration_rate):
        """
        Selects an action by exploration or exploitation.

        Args:
            state (tensor): Batch of images sampled from replay memory.
            exploration_rate (float): Rate of exploration
        """
        self.step += 1
        if exploration_rate < random.random():
            return torch.tensor([[random.randrange(self.n_actions)]])
        else:
            with torch.no_grad():
                return self.policy(state)
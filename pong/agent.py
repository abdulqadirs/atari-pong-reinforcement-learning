import random
import torch
import logging
from config import Config

logger = logging.getLogger("pong")

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
    
    def select_action(self, state, exploration_rate):
        """
        Selects an action by exploration or exploitation.

        Args:
            state (tensor): Batch of images sampled from replay memory.
            exploration_rate (float): Rate of exploration.
        """
        self.step += 1
        if exploration_rate < random.random():
            return torch.tensor([[random.randrange(self.n_actions)]])
        else:
            with torch.no_grad():
                actions = self.policy.forward(state).max(1)[1].view(-1, 1) 
                logger.info("Agents action shape: {}".format(actions.shape))
                return actions
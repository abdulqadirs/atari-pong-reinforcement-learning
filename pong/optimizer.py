import torch

def adam_optimizer(policy_net, learning_rate):
    """
    Returns the Adam Optimizer.
    Args:
        policy_net (object): CNN to convert state to action.
        learning_rate (float): Step size of optimizer.
    
    Returns:
        The Adam Optimizer.
    """
    params = list(policy_net.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    return optimizer
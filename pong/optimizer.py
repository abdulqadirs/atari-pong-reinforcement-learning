import torch

def adam_optimizer(policy_net, learning_rate):
    """
    Returns the Adam Optimizer.
    Args:
        encoder (object): Image encoder (CNN).
        decoder (object): Image decoder (LSTM).
        learning_rate (float): Step size of optimizer.
    
    Returns:
        The Adam Optimizer.
    """
    params = list(policy_net.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    return optimizer
import torch.nn as nn


def l1_loss(state_action_values, expected_state_action_values):
    """
    Calculate mean absolute error between each element in predicted and target state action values.
    Args:
        state_action_values (tensor):
        target_state_action_values (tensor):
    
    Returns:
        l1_loss (float):
    """
    loss_function = nn.L1Loss()
    loss = loss_function(state_action_values, expected_state_action_values)

    return loss

def mse_loss(state_action_values, expected_state_action_values):
    """
    Measure the mean squared error between each element of predicted and target state action values.
    Args:
        state_action_values (tensor):
        expected_state_action_values (tensor):
    
    Returns:
        mse_loss (float):
    """
    loss_function = nn.MSELoss()
    loss = loss_function(state_action_values, expected_state_action_values)

    return loss
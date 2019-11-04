import math

def get_exploration_rate(eps_start, eps_end, eps_decay, steps_done):

    return eps_end + (eps_start - eps_end) * math.exp(-1 * steps_done / eps_decay   )

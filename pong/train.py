from collections import namedtuple
import logging
from pathlib import Path
from itertools import count
import torch

from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.read_config import reading_config
from utils.setup_logging import setup_logging
from environment import make_env
from policies.resnet_policy import Resnet18
from epsilon_greedy_strategy import get_exploration_rate
from agent import Agent
from replay_memory import ReplayMemory
from config import Config
from optimizer import adam_optimizer
from loss_functions import l1_loss, mse_loss

logger = logging.getLogger('pong')

def main():

     #output directory
    output_dir = Path('output')
    output_dir.mkdir(parents=True, exist_ok=True)

    #setup logging
    logfile_path = Path(output_dir / Config.get('logfile'))
    setup_logging(logfile=logfile_path)

    #read config file
    config_file = Path("../config.ini")
    reading_config(config_file)

    #environment
    env_name = Config.get("env_name")
    env = make_env(env_name)

    #configs
    batch_size = Config.get("training_batch_size")
    episodes = Config.get("episodes")
    gamma = Config.get("gamma")
    learning_rate = Config.get("learning_rate")
    epsilon_start = Config.get("epsilon_start")
    epsilon_end = Config.get("epsilon_end")
    epsilon_decay = Config.get("epsilon_decay")
    feature_extraction = Config.get("feature_extraction")
    n_actions = env.action_space.n
    device = Config.get("device")

    #policy network
    policy_network = Resnet18(n_actions, feature_extraction).to(device)
    #target network
    target_network = Resnet18(n_actions, feature_extraction).to(device)
    #initializing the weights of target network
    target_network.load_state_dict(policy_network.state_dict())
    #freezing the target network
    target_network.eval()

    #optimizer
    optimizer = adam_optimizer(policy_network, learning_rate)

    #loss function
    criterion = l1_loss

    #agent
    agent = Agent(policy_network, n_actions)

    #experience
    Experience = namedtuple('Experience',('state', 'action', 'reward', 'next_state'))
    memory_size = Config.get("memory_size")
    memory = ReplayMemory(memory_size)

    for episode in range(episodes):
        state = env.reset()
        exploration_rate = get_exploration_rate(epsilon_start, epsilon_end, epsilon_decay, episode)
        for timestep in count():
            # obs = env.render(mode = 'rgb_array')
            action = agent.select_action(state, exploration_rate).to(device)
            observation, reward, done, info = env.step(action.item())
            reward = torch.tensor([reward], device = device)

            old_state = state
            new_state = observation
            if not done:
                next_state = new_state - old_state
            else:
                next_state = None
            experience = Experience(state, action, reward, next_state)
            memory.push(experience)
            state = next_state

            #sampling from the memory
            batch = memory.sample(batch_size)
            batch = Experience(*zip(*batch))
            state_batch = torch.cat(batch.state, 0)
            action_batch = torch.cat(batch.action, 0)
            reward_batch = torch.cat(batch.reward, 0)

            state_action_values = policy_network(state_batch).gather(1, action_batch)

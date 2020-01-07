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
from policies.alexnet_policy import Alexnet
from agent import Agent
from replay_memory import ReplayMemory
from config import Config
from optimizer import adam_optimizer
from loss_functions import l1_loss, mse_loss
from atari_pong import Pong

logger = logging.getLogger('pong')

def main():

    #output directory
    #output_dir = Path('/content/drive/My Drive/atari-pong-reinforcement-learning/output')
    output_dir = Path("../output")
    output_dir.mkdir(parents=True, exist_ok=True)

    #setup logging
    logfile_path = Path(output_dir / "output.log")
    setup_logging(logfile=logfile_path)

    #read config file
    #config_file = Path('/content/drive/My Drive/atari-pong-reinforcement-learning/config.ini')
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
    target_update = Config.get("target_update")

    #policy network
    #policy_network = Resnet18(n_actions, feature_extraction).to(device)
    policy_network = Alexnet(n_actions, feature_extraction).to(device)
    #target network
    # target_network = Resnet18(n_actions, feature_extraction).to(device)
    target_network = Alexnet(n_actions, feature_extraction).to(device)
    #initializing the weights of target network
    target_network.load_state_dict(policy_network.state_dict())
    #freezing the target network's weights
    target_network.eval()

    #optimizer
    optimizer = adam_optimizer(policy_network, learning_rate)

    #loss function
    criterion = l1_loss

    #experience
    #Experience = namedtuple('Experience',('state', 'action', 'reward', 'next_state'))
    memory_size = Config.get("memory_size")
    memory = ReplayMemory(memory_size)

    #loading the checkpoint
    checkpoint_file = Path(output_dir / Config.get("checkpoint_file"))
    checkpoint_pong = load_checkpoint(checkpoint_file)
    start_episode = 1
    if checkpoint_pong is not None:
        start_episode = checkpoint_pong['episode'] + 1
        policy_network.load_state_dict(checkpoint_pong['policy_net'])
        optimizer.load_state_dict(checkpoint_pong['optimizer'])
    del checkpoint_pong

    #agent
    agent = Agent(policy_network, n_actions)

    #model
    model = Pong(env, policy_network, target_network, agent, optimizer, criterion, memory, output_dir)

    #training
    #model.train(episodes, target_update, start_episode, batch_size, epsilon_start, epsilon_end, epsilon_decay, gamma)
    model.evalutate()


if __name__ == "__main__":
    main()
#taken from pytorch Reinforcement Learning (DQN) tutorial
#https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

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
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    #setup logging
    logfile_path = Path(output_dir / "output.log")
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
    target_update = Config.get("target_update")

    #policy network
    policy_network = Resnet18(n_actions, feature_extraction).to(device)
    #target network
    target_network = Resnet18(n_actions, feature_extraction).to(device)
    #initializing the weights of target network
    target_network.load_state_dict(policy_network.state_dict())
    #freezing the target network's weights
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

    #loading the checkpoint
    checkpoint_file = Path(output_dir / Config.get("checkpoint_file"))
    checkpoint_pong = load_checkpoint(checkpoint_file)
    start_episode = 1
    if checkpoint_pong is not None:
        start_episode = checkpoint_pong['episode'] + 1
        policy_network.load_state_dict(checkpoint_pong['policy_net'])
        optimizer.load_state_dict(checkpoint_pong['optimizer'])
    del checkpoint_pong

    for episode in range(start_episode, episodes + 1):
        state = env.reset()
        exploration_rate = get_exploration_rate(epsilon_start, epsilon_end, epsilon_decay, episode)
        print("Episode: ", episode)
        logger.info("Episode: {}".format(episode))
        for timestep in count():
            # obs = env.render(mode = 'rgb_array')
            state = state.to(device)
            action = agent.select_action(state, exploration_rate).to(device)
            observation, reward, done, info = env.step(action.item())
            reward = torch.tensor([reward], device = device)
            
            #storing the difference of states in the memory.
            old_state = state
            new_state = observation.to(device)
            if not done:
                next_state = (new_state - old_state).to(device)
            else:
                next_state = None

            experience = Experience(state, action, reward, next_state)
            memory.push(experience)
            state = next_state.to(device)

            #sampling from the memory
            current_memory_size = memory.get_size()
            if current_memory_size >= batch_size:
                batch = memory.sample(batch_size)
                batch = Experience(*zip(*batch))

                #final state: last state of an episode.
                #creates a mask (list of booleans) of 'next_states' i.e [False, True, True, False, True]
                #False/0 means that the corresponding 'next_state' of batch is final state of episode.
                #True/1 means that thae corresponding 'next_state' of batch is not final state of episode.
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
                #concatenating the non final 'next_states'
                non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

                state_batch = torch.cat(batch.state, 0)
                action_batch = torch.cat(batch.action, 0)
                reward_batch = torch.cat(batch.reward, 0)

                #policy network: calculates the q_values for the given batch of states
                #actions_batch: actions taken by agent for the given batch in the past.
                #gather(): selects the q_values of actions that would've been taken using actions_batch
                state_action_values = policy_network(state_batch).gather(1, action_batch)

                #masking  q_values of the final 'next_state' with zeros 
                next_state_values = torch.zeros(batch_size, device=device)
                next_state_values[non_final_mask] = target_network(non_final_next_states).max(1)[0].detach()

                #expected q value = reward + (gamma * next_q_values) 
                expected_state_action_values = reward_batch + (next_state_values * gamma)
                expected_state_action_values = expected_state_action_values.unsqueeze(1)
            
                #loss
                loss = criterion(state_action_values, expected_state_action_values)

                #optimization
                optimizer.zero_grad()
                loss.backward()
                for param in policy_network.parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(-1, 1)
                optimizer.step()

            if done:
                logger.info("Episode {} completed.".format(episode))
                break
        #updating the weights of target network
        #saving the checkpoint of policy network
        if episode % target_update == 0:
            target_network.load_state_dict(policy_network.state_dict())
            save_checkpoint(episode = episode,
                            outdir = output_dir,
                            policy_net = policy_network,
                            optimizer = optimizer,
                            criterion = criterion)
        

if __name__ == "__main__":
    main()
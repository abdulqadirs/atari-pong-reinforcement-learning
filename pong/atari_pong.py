# some code has been taken from pytorch Reinforcement Learning (DQN) tutorial
#https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import logging
import torch
from collections import namedtuple
from itertools import count

from config import Config
from utils.checkpoint import save_checkpoint
from epsilon_greedy_strategy import get_exploration_rate

logger = logging.getLogger('pong')

class Pong:
    """
    Runs the model for training and evaluating the pretrained embeddings.

    Attributes:
        policy_network (object): 
        target_network (object):
        agent (object): 
        optimizer: adam optimizer..
        criterion: loss function
        replay_memory (list): 
        output_dir (Path): path of output directory

    """
    def __init__(self, environment, policy_network, target_network, agent,
                 optimizer, criterion, replay_memory, output_dir):
        self.environment = environment
        self.policy_network = policy_network
        self.target_network = target_network
        self.agent = agent
        self.optimizer = optimizer
        self.criterion = criterion
        self.memory = replay_memory
        self.output_dir = output_dir
        self.Experience = namedtuple('Experience',('state', 'action', 'reward', 'next_state'))
        self.device = Config.get("device")
    
    def train(self, episodes, target_update, start_episode, batch_size,
                epsilon_start, epsilon_end, epsilon_decay, gamma):
        """
        Training the model.

        Args:
            episodes (int): Total episodes for training.
            target_update (int): Update the target network after given episodes.
            start_episode (int):
            batch_size (int):
            epsilon_start (float):
            epsilon_end (float):
            epsilon_decay (float):
            gamma (float):
        """ 
        print(start_episode)
        for episode in range(start_episode, episodes + 1):
            state = self.environment.reset()
            exploration_rate = get_exploration_rate(epsilon_start, epsilon_end, epsilon_decay, episode)
            print("Episode: ", episode)
            logger.info("Episode: {}".format(episode))
            for timestep in count():
                # obs = env.render(mode = 'rgb_array')
                state = state.to(self.device)
                action = self.agent.select_action(state, exploration_rate).to(self.device)
                observation, reward, done, info = self.environment.step(action.item())
                reward = torch.tensor([reward], device = self.device)
                
                #storing the difference of states in the memory.
                old_state = state
                new_state = observation.to(self.device)
                if not done:
                    next_state = (new_state - old_state).to(self.device)
                else:
                    next_state = None

                experience = self.Experience(state, action, reward, next_state)
                self.memory.push(experience)
                if next_state is not None:
                    state = next_state.to(self.device)
                else:
                    state = next_state

                #sampling from the memory
                current_memory_size = self.memory.get_size()
                if current_memory_size >= batch_size:
                    batch = self.memory.sample(batch_size)
                    batch = self.Experience(*zip(*batch))

                    #final state: last state of an episode.
                    #creates a mask (list of booleans) of 'next_states' i.e [False, True, True, False, True]
                    #False/0 means that the corresponding 'next_state' of batch is final state of episode.
                    #True/1 means that thae corresponding 'next_state' of batch is not final state of episode.
                    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                                     dtype=torch.bool).to(self.device)
                    #concatenating the non final 'next_states'
                    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(self.device)

                    state_batch = torch.cat(batch.state, 0)
                    action_batch = torch.cat(batch.action, 0)
                    reward_batch = torch.cat(batch.reward, 0)

                    #policy network: calculates the q_values for the given batch of states
                    #actions_batch: actions taken by agent for the given batch in the past.
                    #gather(): selects the q_values of actions that would've been taken using actions_batch
                    state_action_values = self.policy_network(state_batch).gather(1, action_batch)

                    #masking  q_values of the final 'next_state' with zeros 
                    next_state_values = torch.zeros(batch_size, device = self.device)
                    next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0].detach()

                    #expected q value = reward + (gamma * next_q_values) 
                    expected_state_action_values = reward_batch + (next_state_values * gamma)
                    expected_state_action_values = expected_state_action_values.unsqueeze(1)
                
                    #loss
                    loss = self.criterion(state_action_values, expected_state_action_values)

                    #optimization
                    self.optimizer.zero_grad()
                    loss.backward()
                    for param in self.policy_network.parameters():
                        if param.grad is not None:
                            param.grad.data.clamp_(-1, 1)
                    self.optimizer.step()

                if done:
                    logger.info("Episode {} completed.".format(episode))
                    break
            #updating the weights of target network
            #saving the checkpoint of policy network
            if episode % target_update == 0:
                self.target_network.load_state_dict(self.policy_network.state_dict())
                save_checkpoint(episode = episode,
                                outdir = self.output_dir,
                                policy_net = self.policy_network,
                                optimizer = self.optimizer,
                                criterion = self.criterion)

    def evalutate(self):
        """
        Runs the model using pretrained embedding in evaluation mode.
        """
        state = self.environment.reset()
        for i in range(1000):
            self.environment.render()
            action = self.policy_network.forward(state).max(1)[1].view(-1, 1).to(self.device)
            observation, reward, done, info = self.environment.step(action.item())
            state = observation.to(self.device)
            # old_state = state
            # new_state = observation.to(self.device)
            # if not done:
            #     next_state = (new_state - old_state).to(self.device)
            # else:
            #     next_state = None
            
            # if next_state is not None:
            #     state = next_state.to(self.device)
            print(reward)
            if done:
                state = self.environment.reset()
        self.environment.close()
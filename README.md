# Playing Atari Pong With Reinforcement Learning
This is the PyTorch implementation of deef reinforcement learning algorithm to play Atari Pong game using [OpenAI Gym](https://gym.openai.com/).

## Setup
Insall the requirements:
```
pip install -r requirements.txt
```

## Usage

```
python3 pong/train.py
```

## Description

### State
A *state* in reinforcement learning is the observation that the agent receives from the environment.

### Policy
A *policy* is the mapping from the perceived *states* of the environment to the actions to be taken when in those states.

### Reward Signal
A *reward signal* is the goal in reinforcement learning. The agent tries to maximize the total *reward* in long run.

### Value Function
The *reward signal* indicates what is good in immediate sense, whereas the *value function* measures what is good in long run. Each state of environment is assigned a *value* which is the total amount of reward an agent is expected to receive, starting from that *state*.

### Model
A *model* in reinforcemnt learning mimics the behavior of the environment.

### Deep Q-Learning Training Process
<ol>
  <li>Target Network: A copy of policy network.</li>
  <li>Initialize the Replay Memory: Used for storing the experience SARS'(state, action, reward, next-state).</li>
  <li>For each episode: </li>
    <ol>
     <li>Reset the environment to get starting state.</li>
     <li>Calculate exploration rate.</li>
     <li>For each time step:
      <ol>
        <li>Select an action using exploration or exploitation.</li>
        <li>Take the action, get reward from the envionment and move to the next-state.</li>
        <li>Store SARS'(state, action, reward, next-state) in the replay memory.</li>
        <li>Sample a batch of data (SARS') from replay memory.</li>
        <li>Preprocess the sampled batch of states.</li>
        <li>Pass the sampled batch of states through policy network to calculate the q-values.</li>
        <li>Calculate the q-values for next-states using target network.</li>
        <li>Calculate: Expected q-values = reward + next-states-q-values * gamma.</li>
        <li>Calculate the loss beteen q-values of policy network and expected q-values.</li>
        <li>Update the weights of policy network to minimize the loss.</li>
      </ol>
      </li>
      <li>After 'u' episodes, update the weights of target network using the weights of policy network.</li>
    </ol>
</ol>

## Todo List
- [ ] Fixing the local optimum problem.
- [ ] Calculating the moving average of scores.
- [ ] Plotting the scores using TensorBoard.

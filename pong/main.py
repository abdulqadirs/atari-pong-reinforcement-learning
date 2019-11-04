from collections import namedtuple

from environment import make_env
from policies.resnet_policy import Resnet18


# env_name = "Pong-v0"
env_name = 'PongNoFrameskip-v4'
env = make_env(env_name)
state = env.reset()
# obs = env.render(mode = 'rgb_array')
print(state.shape)
n_actions = env.action_space.n
feature_extraction = True
resnet = Resnet18(n_actions, feature_extraction)
actions = resnet.forward(state)
print("actions: ", actions)


Experience = namedtuple('Experience',('state', 'action', 'reward', 'next_state'))
e = Experience(1, 2 , 3, 4)

#print(env.action_space.n)
#print(env.unwrapped.get_action_meanings())



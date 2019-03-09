import gym
import math
import torch
import random
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.Functional as F
import torchvision.transforms as T

from utils import *
from models import *

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9 # Decays epsillon for exploration/exploatation
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
T_MAX = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('CartPole-v0').unwrapped # Setup environment
env.reset() 
init_screen = get_screen(env)
_, _, screen_height, screen_width = init_screen.shape # Ignored Batch, Channel

policy_net = DQN(screen_height, screen_width).to(device)
target_net = DQN(screen_height, screen_width).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

loss_fn = nn.MSELoss()
optimizer = torch.optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)
steps_done = 0
num_episodes = 50

def select_action(state):
    ''' Acts EPS-Greedily on Policy-Network '''
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if(sample > eps_threshold):
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net.max(1)[1].view(1, 1)
    else:
        torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)

def optimize_model():
    ''' Trains the network on a single batch from ER '''
    if len(memory) < BATCH_SIZE:
        return 
    transitions = memory.sample(BATCH_SIZE)
    batch = *zip(*transitions)

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch[2])), device=device, dtype=torch.uint8) # batch[2] == next_state
    non_final_next_states = torch.cat([s for s in batch[2] if s is not None])
    state_batch = torch.cat(batch[0]) # Gets all the states from the batches, etc.
    action_batch = torch.cat(batch[1])
    reward_batch = torch.cat(batch[3])

    state_action_values = policy_net(state_batch).gather(1, action_batch) 

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def train():
    for i in range(0, num_episodes):
        env.reset()
        last_screen = get_screen(env)
        current_screen = get_screen(env)
        state = current_screen - last_screen

        for t in range(0, T_MAX):
            action = select_action()
            _, reward, done, _  = env.step(action)
            reward = torch.tensor([reward], device=device)

            last_screen = current_screen
            current_screen = get_screen()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None


            memory.push((state, action, next_state, reward))
            state = next_state
            optimize_model()

            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
    print('Complete')
    env.render()
    env.close()

if __name__ == '__main__':
    train()
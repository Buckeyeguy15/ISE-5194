# -*- coding: utf-8 -*-
"""

# Importing

If you are a Mac user:
"""

'''!apt-get install x11-utils
'''
"""Whether you are a Windows or Mac user, you will need the following code blocks:"""
'''
# Commented out IPython magic to ensure Python compatibility.
!pip install gym
!apt-get install python-opengl -y
!apt install xvfb -y

# Special gym environment
!pip install gym[atari]

# For rendering environment, you can use pyvirtualdisplay.
!pip install pyvirtualdisplay
!pip install piglet
'''
# To activate virtual display 
# need to run a script once for training an agent as follows
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()

# This code creates a virtual display to draw game images on. 
# If you are running locally, just ignore it
import os
'''
if type(os.environ.get("DISPLAY")) is not str or len(os.environ.get("DISPLAY"))==0:
    !bash ../xvfb start
#     %env DISPLAY=:1
'''
# Commented out IPython magic to ensure Python compatibility.

# Import libraries


import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(40) # error only
#import tensorflow as tf
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
import math
import glob
import io
import base64
from IPython.display import HTML

from IPython import display as ipythondisplay

"""
Utility functions to enable video recording of gym environment and displaying it
To enable video, just do "env = wrap_env(env)""
"""

def show_video():
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
  else: 
    print("Could not find video")
    

def wrap_env(env):
  env = Monitor(env, './video', force=True)
  return env

"""# Problem 1"""

# Specify which environment to use.
env = gym.make("Taxi-v3").env
env.reset()

# Initialize table of Q-values
# Hint: to access a specific value in the q_table, do this:
#            q_table[state, action]
q_table = np.zeros([env.observation_space.n, env.action_space.n])

##########################################
# Initialize RL Parameters
##########################################


# For plotting metrics
cumulative_reward_each_episode = []
epsilon_each_episode = []

# For each episode
maxNumEpisodes = 2000
for i in range(maxNumEpisodes):

  # Reset to initial conditions
  state = env.reset()

  # The variable 'cumulative_reward' will store the sum of the accumulated 
  # reward for an entire episode. Set this value to zero at the start of each 
  # episode.
  cumulative_reward = 0
  done = False

  # While the episode is not finished
  while not done:

    ##########################################
    # For every time step, using epsilon-greedy to choose between
    # exploration and exploitation.
    # Implement epsilon-greedy exploration.
    # Hint: to return a random action, do this:
    #           action = env.action_space.sample()
    ##########################################


    # Take the action.
    # This moves the agent to a new state and earns a reward
    next_state, reward, done, info = env.step(action)

    # Add the reward just earned to the cumulative reward variable
    cumulative_reward += reward


    ##########################################
    # Update your estimate of Q(s,a)
    # Hint: to access a specific value in the q_table, do this:
    #            q_table[state, action]
    ##########################################


    # Set your state variable to next_state for the next loop.
    state = next_state

    # If this episode is finished, take care of a few things:
    if done:
      # Save the cumulative reward from the previous episode to an array.
      cumulative_reward_each_episode.append(cumulative_reward)

      # Save the epsilon used in this episode.
      epsilon_each_episode.append(epsilon)

      ##########################################
      # Decay epsilon,
      # If you want to decay or change the value of epsilon at the end of
      # each episode, do so here.
      ##########################################


  if i % 100 == 0:
    print('Episode: {0}'.format(i)) 

print("Training finished.\n")

# Plot the Cumulative Reward and Epsilon value through time.
fsize = 15

plt.plot(cumulative_reward_each_episode)
plt.title('Cumulative Reward through Time', fontsize=fsize)
plt.xlabel('Episode', fontsize=fsize)
plt.ylabel('Cumulative Reward', fontsize=fsize)
plt.show() 

plt.plot(epsilon_each_episode)
plt.title('Exploration (epsilon) through Time', fontsize=fsize)
plt.xlabel('Episode', fontsize=fsize)
plt.ylabel('epsilon', fontsize=fsize)
plt.show()

# Once training is finished, run through one more episode using exploitation only. 

done = False

state = env.reset()

while True:
  env.render()

  ##########################################
  # Choose an action based on exploitation.
  ##########################################
  action = state, reward, done, info = env.step(action)
   
  if done: 
    break
env.close()

"""# Problem 2"""

# I have included code to help you discretize the state space.
# You DO NOT need to keep these specific bin ranges.
# In fact, you may not want to keep these bin ranges.
# I have provided this code to make it easier for you to modify and to save
# you time.
# You can alter or discretize the state space however you wish.
# You do not need to keep all 4 state features if you have an argument for 
# eliminating features.

import pandas as pd

# Discretize input state to make Q-table and to reduce dimensionality
def discretize(state):

  #print ( state )

  # First, set up arrays of the left bin edges
  # Note: your bin sizes do not need to be of uniform width.
  bins_pos = [-4.8, -1, 1, 4.8]
  bins_vel = [-3.4*10**38, -5, 5, 3.4*10**38 ]
  bins_w = [-3.4*10**38, -100, -50, -40, -30, -20, -10, -5, -2, 0, 2, 5, 10, 20, 30, 40, 50, 100, 3.4*10**38]

  angle_range = 0.43*2
  num_angle_bins = 12
  angle_bin_stepsize = angle_range / num_angle_bins
  bins_ang = [-0.43]
  for i in range(1, num_angle_bins):
    bins_ang.append( bins_ang[i-1]+ angle_bin_stepsize )

  cart_position_bin = pd.cut([state[0]], bins=bins_pos, include_lowest=True)
  cart_velocity_bin = pd.cut([state[1]], bins=bins_vel, include_lowest=True)
  pole_angle_bin    = pd.cut([state[2]], bins=bins_ang, include_lowest=True)
  angle_rate_bin    = pd.cut([state[3]], bins=bins_w  , include_lowest=True)

  # To verify the order of the state variables:
  #   https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

  return [cart_position_bin[0].left, cart_velocity_bin[0].left, pole_angle_bin[0].left, angle_rate_bin[0].left]

# Simple code to test your discretization.

# Specify which environment to use.
env = gym.make('CartPole-v0')
state = env.reset()

action = env.action_space.sample() # Explore action space
state, reward, done, info = env.step(action)
print ( 'Continuous state: ')
print ( state )

discretized_state = discretize(state)
print ( 'Discretized state: ')
print ( discretized_state )

env.close()

# Specify which environment to use.
env = gym.make('CartPole-v0')
#env = wrap_env(env) 
state = env.reset()

##########################################
# Initialize your Q-values.
# Note: you may use whichever data structure you wish.
#       I used a dictionary, but a list works, too.
##########################################



##########################################
# Initialize RL Parameters
##########################################



# For plotting metrics
cumulative_reward_each_episode = []
epsilon_each_episode = []


# To start off wish debugging your code, use 1 episode. Increase this once
# your code starts to work.
maxNumEpisodes = 1

# For each episode
for i in range(maxNumEpisodes):

  # Reset to initial conditions
  state = env.reset()

  ##########################################
  # Discretize the state
  # Note: you'll need to modify the discretize function
  #       provided above.
  ##########################################
  state = discretize(state)

  # At the beginning of each episode, set the cumulative reward variable to zero.
  cumulative_reward = 0
  done = False

  # For every step in the episode
  while not done:
    #env.render()

    ##########################################
    # For every time step, using epsilon-greedy to choose between
    # exploration and exploitation.
    # Implement epsilon-greedy exploration.
    # Hint: to return a random action, do this:
    #           action = env.action_space.sample()
    ##########################################



    # Take the action.
    # This moves the agent to a new state and earns a reward
    next_state, reward, done, info = env.step(action)

    # Discrete the state
    next_state = discretize(next_state)

    # Add the reward just earned to the cumulative reward variable
    cumulative_reward += reward

    ##########################################
    # Update your estimate of Q(s,a)
    ##########################################



    state = next_state

    # If the episode is finished, do a few things.
    if done:
      # Save the cumulative reward from the previous episode to an array.
      cumulative_reward_each_episode.append(cumulative_reward)

      # Save the epsilon used in this episode.
      epsilon_each_episode.append(epsilon)

      ##########################################
      # Decay epsilon.
      # If you want to decay or change the value of epsilon at the end of
      # each episode, do so here.
      ##########################################


  if i % 100 == 0:
    print('Episode: {0}'.format(i)) 

env.close()
#show_video()
print("Training finished.\n")


# Plot the Cumulative Reward and Epsilon value through time.
fsize = 15

plt.plot(cumulative_reward_each_episode)
plt.title('Cumulative Reward through Time', fontsize=fsize)
plt.xlabel('Episode', fontsize=fsize)
plt.ylabel('Cumulative Reward', fontsize=fsize)
plt.show() 

plt.plot(epsilon_each_episode)
plt.title('Exploration (epsilon) through Time', fontsize=fsize)
plt.xlabel('Episode', fontsize=fsize)
plt.ylabel('epsilon', fontsize=fsize)
plt.show()

# Once training is finished, run through one more episode using exploitation only.

# Discretize input state to make Q-table and to reduce dimensionality
def discretize(state):

  #print ( state )
  # -3.4*10**38, , 3.4*10**38 
  # First, set up arrays of the left bin edges
  # Note: your bin sizes do not need to be of uniform width.
  bins_w = [-40, -30, -20, -10, -5, -2, 0, 2, 5, 10, 20, 30, 40]

  angle_range = 0.43*2
  num_angle_bins = 12
  angle_bin_stepsize = angle_range / num_angle_bins
  bins_ang = [-0.43]
  for i in range(1, num_angle_bins):
    bins_ang.append( bins_ang[i-1]+ angle_bin_stepsize )

  pole_angle_bin    = pd.cut([state[2]], bins=bins_ang, include_lowest=True)
  angle_rate_bin    = pd.cut([state[3]], bins=bins_w  , include_lowest=True)
  
  

  # To verify the order of the state variables:
  #   https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

  return tuple([0, 0, int(pole_angle_bin[0].left), int(angle_rate_bin[0].left)])

# Specify which environment to use.
env = gym.make('CartPole-v0')
#env = wrap_env(env) 
state = env.reset()

##########################################
# Initialize your Q-values.
# Note: you may use whichever data structure you wish.
#       I used a dictionary, but a list works, too.
##########################################

# number of bins for each area of the state space
bins = (1, 1, 6, 13) 

q_table = np.zeros(bins + (2,))

print(np.shape(q_table))

##########################################
# Initialize RL Parameters
##########################################
alpha = 0.5
gamma = 1
epsilon = 0.95
epsilon_min = 0.05

# For plotting metrics
cumulative_reward_each_episode = []
epsilon_each_episode = []


# To start off wish debugging your code, use 1 episode. Increase this once
# your code starts to work.
maxNumEpisodes = 500

# For each episode
for i in range(maxNumEpisodes):

  # Reset to initial conditions
  state = env.reset()

  ##########################################
  # Discretize the state
  # Note: you'll need to modify the discretize function
  #       provided above.
  ##########################################
  state = discretize2(state)

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
    num = random.random()
    action = env.action_space.sample()
    
    if num < epsilon:
      action = env.action_space.sample()

    else:
      action = np.argmax(q_table[state])
    

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
    
    old_val = q_table[state][action]
    next_max = np.max(q_table[next_state])

    new_val = (1-alpha)* old_val + alpha * (reward + gamma * next_max)

    #update table
    q_table[state, action] = new_val
    
    

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
      epsilon = max(epsilon*.99, epsilon_min)

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

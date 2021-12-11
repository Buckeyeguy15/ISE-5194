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
alpha = 0.5
gamma = 0.6
epsilon = 0.95

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

  action_num = 0
  # While the episode is not finished
  while not done:

    # for final episode, print the actions
    
    if i == maxNumEpisodes - 1:
      print('action number: ', action_num)
      action_num += 1
      env.render() 


    ##########################################
    # For every time step, using epsilon-greedy to choose between
    # exploration and exploitation.
    # Implement epsilon-greedy exploration.
    # Hint: to return a random action, do this:
    #           action = env.action_space.sample()
    ##########################################
    num = random.random()
    
    if num < epsilon:
      action = env.action_space.sample()

    else:
      action = np.argmax(q_table[state])

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
    old_val = q_table[state, action]
    next_max = np.max(q_table[next_state])

    new_val = (1-alpha)* old_val + alpha * (reward + gamma * next_max)

    #update table
    q_table[state, action] = new_val

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
      epsilon = epsilon*0.95

  if i % 100 == 0:
    print('Episode: {0}'.format(i)) 

print("Training finished.\n")

# Plot the Cumulative Reward and Epsilon value through time.
fsize = 15

# print('Max reward: ', max(cumulative_reward_each_episode))

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

# print out 1 iteration of the algorithm
env = wrap_env(env)
show_video()
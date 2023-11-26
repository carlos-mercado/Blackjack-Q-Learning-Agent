import gym
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

env = gym.make("Blackjack-v1")

class BlackjackPlayer:
    def __init__(self, learningRate, discountFactor): #constructor 
        self.qValues = defaultdict(lambda: np.zeros(env.action_space.n))
        self.learningRate = learningRate
        self.discount = discountFactor
    
    def get_action(self, observation):
        """
        Return best action from given obeservation, but only
        80 percent of the time to maximize our chances of properly
        exporing the environment

        """
        
        chance = 0.8
        if np.random.random() < (1-chance): #if the random number falls between [0.0, 0.2), try to explore
            return env.action_space.sample()
        else:
            return int(np.argmax(self.qValues[observation])) 
    
    def update(self, observation, action, reward, terminated, nextObs):
        
        if (terminated): #game is finished
            future_qValue = 0
        else:#max(a') Qk(s', a')
            future_qValue = np.max(self.qValues[nextObs])
        
        sample = reward + (self.learningRate * future_qValue)

        #incorporate the new estimate into a running average
        #Qk+1(s, a) <- Qk(s, a) + learning_rate * sample
        self.qValues[observation][action] = self.qValues[observation][action] + self.learningRate * (sample - self.qValues[observation][action])

class RandomMoveAgent:
    def __init__(self, learningRate, discountFactor): #constructor 
        self.qValues = defaultdict(lambda: np.zeros(env.action_space.n))
        self.learningRate = learningRate
        self.discount = discountFactor
    
    def get_action(self, observation):
        """
        Return random action
        """
        return env.action_space.sample()
    
    def update(self, observation, action, reward, terminated, nextObs):
        
        if (terminated): #game is finished
            future_qValue = 0
        else:#max(a') Qk(s', a')
            future_qValue = np.max(self.qValues[nextObs])
        
        sample = reward + (self.learningRate * future_qValue)

        #incorporate the new estimate into a running average
        #Qk+1(s, a) <- Qk(s, a) + learning_rate * sample
        self.qValues[observation][action] = self.qValues[observation][action] + self.learningRate * (sample - self.qValues[observation][action])


#TRAINING
episodes = 200000

player = BlackjackPlayer(learningRate=0.01, discountFactor=0.95)
randomPlayer = RandomMoveAgent(learningRate=0.01, discountFactor=0.95)

"""

env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=episodes)
wins = 0
wrs = []
xs = []
for episode in tqdm(range(episodes)):
    obs, _ = env.reset()
    done = False

    while not done:
        action = player.get_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        if reward == 1:
            wins += 1
        if episode % 1000 == 0 and episode > 0:
            wrs.append(wins/episode)
            xs.append(episode)

        player.update(obs, action, reward, terminated, next_obs)

        done = terminated or truncated

        obs = next_obs

# Plotting the points as a line graph
plt.plot(xs, wrs, marker='o', linestyle='-')  # 'o' for markers, '-' for line
plt.title('Win Rate Over Episodes')
plt.xlabel('Iteration')
plt.ylabel('Win Rate')
plt.grid(True)  # Adding grid lines for better readability
plt.show()

    """    
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=episodes)
wins = 0
wrs = []
xs = []
for episode in tqdm(range(episodes)):
    obs, _ = env.reset()
    done = False

    while not done:
        action = randomPlayer.get_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        if reward == 1:
            wins += 1
        if episode % 1000 == 0 and episode > 0:
            wrs.append(wins/episode)
            xs.append(episode)


        done = terminated or truncated

        obs = next_obs


# Plotting the points as a line graph
plt.plot(xs, wrs, marker='o', linestyle='-')  # 'o' for markers, '-' for line
plt.title('Win Rate Over Episodes (Random Policy)')
plt.xlabel('Iteration')
plt.ylabel('Win Rate')
plt.grid(True)  # Adding grid lines for better readability
plt.show()
def createPolicyMatrix(player):
    #Matrix is in this form: https://www.techopedia.com/wp-content/uploads/2023/04/TECHOPEDIA-DEALERS-CARD-TABLE.png

    #row = player hand
    #column = dealer face up card.

    policyMatrixUsableAce = defaultdict(lambda: defaultdict(int))
    policyMatrixNoAce = defaultdict(lambda: defaultdict(int))

    
    #hard total policy construction(no ace)
    hardTotalPolicyMatrix = np.zeros((18,10)) 
    for obs, action_values in player.qValues.items():
        if(obs[0] == 11):
            somethin = "fjkdslaj"

        if(obs[2] == False):
            standOrHit = np.argmax(action_values)
            hardTotalPolicyMatrix[obs[0] - 4][obs[1] - 1] = standOrHit

    # Create a mapping of values in the matrix to blackjack actions and colors
    blackjack_actions = {
        0: ('Stand', 'red'),
        1: ('Hit', 'green'),
        # Add more actions and their corresponding colors as needed
    }

    # Initialize lists to store actions and colors
    actions = []
    colors = []

    # Iterate through the matrix and extract actions and colors
    for row in hardTotalPolicyMatrix[:-1]:
        action_row = []
        color_row = []
        for cell_value in row:
            action, color = blackjack_actions[cell_value]
            action_row.append(action)
            color_row.append(color)
        actions.append(action_row)
        colors.append(color_row)

    # Display the matrix as a table-like representation
    fig, ax = plt.subplots()
    ax.axis('off')
    table = ax.table(cellText=actions, loc='center', cellLoc='center', colLabels=range(1, 11), rowLabels=range(4, 21), cellColours=colors)

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1)  # Adjust the scaling of the table

    plt.title('Blackjack Hard Policy Chart')
    plt.show()

    #soft total policy construction (ace)
    softPolicy = np.zeros((10,10)) 
    for obs, action_values in player.qValues.items():
        if(obs[2] == True):
            standOrHit = np.argmax(action_values)
            softPolicy[obs[0] - 12][obs[1] - 1] = standOrHit
            
    # Create a mapping of values in the matrix to blackjack actions and colors
    blackjack_actions = {
        0: ('Stand', 'red'),
        1: ('Hit', 'green'),
        # Add more actions and their corresponding colors as needed
    }

    # Initialize lists to store actions and colors
    actions = []
    colors = []

    # Iterate through the matrix and extract actions and colors
    for row in softPolicy[:-1]:
        action_row = []
        color_row = []
        for cell_value in row:
            action, color = blackjack_actions[cell_value]
            action_row.append(action)
            color_row.append(color)
        actions.append(action_row)
        colors.append(color_row)

    # Display the matrix as a table-like representation
    fig, ax = plt.subplots()
    ax.axis('off')
    table = ax.table(cellText=actions, loc='center', cellLoc='center', colLabels=range(1, 11), rowLabels=range(12, 21), cellColours=colors)

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1)  # Adjust the scaling of the table

    plt.title('Blackjack Soft Policy Chart')
    plt.show()

#createPolicyMatrix(player)
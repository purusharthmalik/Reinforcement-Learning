import numpy as np
import gym
from gym import spaces

class SimpleGridWorld(gym.Env):
    def __init__(self, grid_size=(5, 5), start=(0, 0), goal=(4, 4)):
        super(SimpleGridWorld, self).__init__()
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.state = start
        
        # Define action space: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)
        
        # Define observation space (grid positions)
        self.observation_space = spaces.Box(low=0, high=max(grid_size)-1, shape=(2,), dtype=np.int32)

    def reset(self):
        self.state = list(self.start)
        return np.array(self.state)

    def step(self, action):
        if action == 0:  # Up
            self.state[0] = max(0, self.state[0] - 1)
        elif action == 1:  # Down
            self.state[0] = min(self.grid_size[0] - 1, self.state[0] + 1)
        elif action == 2:  # Left
            self.state[1] = max(0, self.state[1] - 1)
        elif action == 3:  # Right
            self.state[1] = min(self.grid_size[1] - 1, self.state[1] + 1)

        done = tuple(self.state) == self.goal
        reward = 1 if done else -0.01  # Small penalty for each step

        return np.array(self.state), reward, done, {}

    def render(self):
        grid = np.zeros(self.grid_size)
        grid[self.goal] = 2  # Goal position
        grid[tuple(self.state)] = 1  # Agent position
        print(grid)

import random

class DynaQAgent:
    def __init__(self, env):
        self.env = env
        self.q_table = np.zeros((*env.grid_size, env.action_space.n))  # Q-values for each state-action pair
        self.model = {}  # Model to store state transitions and rewards
        self.alpha = 0.1   # Learning rate
        self.gamma = 0.9   # Discount factor
        self.epsilon = 0.1  # Exploration rate

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            return np.argmax(self.q_table[state[0], state[1]])  # Exploit

    def learn_from_interaction(self, state, action, reward, next_state):
        # Update Q-value using the Bellman equation
        best_next_action = np.argmax(self.q_table[next_state[0], next_state[1]])
        td_target = reward + self.gamma * self.q_table[next_state[0], next_state[1], best_next_action]
        td_delta = td_target - self.q_table[state[0], state[1], action]
        
        # Update Q-value with learning rate
        self.q_table[state[0], state[1], action] += self.alpha * td_delta
        
        # Update the model with the new experience (state transition and reward)
        if (tuple(state), action) not in self.model:
            self.model[(tuple(state), action)] = (next_state, reward)
    
    def plan(self):
        for _ in range(10):  # Number of planning steps
            state_action_pair = random.choice(list(self.model.keys()))
            state, action = state_action_pair
            
            next_state, reward = self.model[state_action_pair]
            next_state_tuple = tuple(next_state)
            
            # Update Q-value based on the model's prediction
            best_next_action = np.argmax(self.q_table[next_state_tuple])
            td_target = reward + self.gamma * self.q_table[next_state_tuple][best_next_action]
            td_delta = td_target - self.q_table[state][action]
            
            # Update Q-value with learning rate
            self.q_table[state][action] += self.alpha * td_delta
    
    def train(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                # Learn from interaction with the real environment
                self.learn_from_interaction(state.tolist(), action, reward, next_state.tolist())
                
                # Plan using the learned model
                self.plan()
                
                state = next_state

if __name__ == "__main__":
    env = SimpleGridWorld()
    agent = DynaQAgent(env)

    agent.train(100)  # Train for a number of episodes

    print("Training complete.")
import numpy as np
import random

class SimpleGridWorld:
    def __init__(self, size=(5, 5), start=(0, 0), goal=(4, 4)):
        self.size = size
        self.start = start
        self.goal = goal
        self.state = start

    def reset(self):
        self.state = self.start
        return self.state
    
    def step(self, action):
        if action == 0:  # Up
            self.state = (max(self.state[0] - 1, 0), self.state[1])
        elif action == 1:  # Down
            self.state = (min(self.state[0] + 1, self.size[0] - 1), self.state[1])
        elif action == 2:  # Left
            self.state = (self.state[0], max(self.state[1] - 1, 0))
        elif action == 3:  # Right
            self.state = (self.state[0], min(self.state[1] + 1, self.size[1] - 1))

        done = self.state == self.goal
        reward = 1 if done else -0.1
        
        return self.state, reward, done

    def render(self):
        grid = np.zeros(self.size)
        grid[self.goal] = 2  # Goal position
        grid[self.state] = 1  # Agent position
        print(grid)

class TD_Agent:
    def __init__(self, env, alpha=0.1, gamma=0.9):
        self.env = env
        self.alpha = alpha   # Learning rate
        self.gamma = gamma   # Discount factor
        self.V = np.zeros(env.size)  # State-value function

    def choose_action(self):
        return random.choice([0, 1, 2, 3])  # Random policy for exploration

    def learn(self):
        state = self.env.reset()
        
        done = False
        while not done:
            action = self.choose_action()
            next_state, reward, done = self.env.step(action)

            # Bootstrapping update for state-value function V(s)
            td_target = reward + self.gamma * self.V[next_state]
            td_delta = td_target - self.V[state]
            self.V[state] += self.alpha * td_delta
            
            state = next_state

if __name__ == "__main__":
    env = SimpleGridWorld()
    agent = TD_Agent(env)

    for episode in range(100):
        agent.learn()

    print("Learned State-Value Function:")
    print(agent.V)


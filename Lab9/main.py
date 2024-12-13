import numpy as np
import random

class OneDGridEnv:
    def __init__(self):
        self.states = [1, 2, 3, 4, 5]
        self.current_state = 1
    
    def reset(self):
        self.current_state = 1
        return self.current_state
    
    def step(self, action):
        # Action: 0 = left, 1 = right
        if action == 0:  # Move left
            if self.current_state > 1:
                self.current_state -= 1
        elif action == 1:  # Move right
            if self.current_state < 5:
                self.current_state += 1
        
        # Reward structure
        if self.current_state == 5:
            reward = 1
            done = True
        else:
            reward = -0.1
            done = False
        
        return self.current_state, reward, done

class TD0Agent:
    def __init__(self, env, alpha=0.1, gamma=0.9):
        self.env = env
        self.alpha = alpha   # Learning rate
        self.gamma = gamma   # Discount factor
        self.V = np.zeros(len(env.states))  # State-value function initialized to zero

    def choose_action(self):
        # Randomly choose to move left or right
        return random.choice([0, 1])  

    def learn(self):
        state = self.env.reset()
        
        done = False
        while not done:
            action = self.choose_action()
            next_state, reward, done = self.env.step(action)
            
            # Update the value function using TD(0)
            td_target = reward + self.gamma * self.V[next_state - 1]  # next_state - 1 for index
            td_delta = td_target - self.V[state - 1]                  # state - 1 for index
            
            # Update the value function V(s)
            self.V[state - 1] += self.alpha * td_delta
            
            state = next_state

if __name__ == "__main__":
    env = OneDGridEnv()
    agent = TD0Agent(env)

    num_episodes = 1000
    for episode in range(num_episodes):  
        agent.learn()

    print("Learned State-Value Function:")
    print(agent.V)

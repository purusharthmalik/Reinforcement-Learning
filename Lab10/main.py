import numpy as np

class SimpleGridEnv:
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

import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)

class PolicyGradientAgent:
    def __init__(self, env):
        self.env = env
        self.policy_net = PolicyNetwork(input_size=1, output_size=2)  # Two actions: left and right
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.01)
    
    def choose_action(self, state):
        state_tensor = torch.tensor([[state]], dtype=torch.float32)
        action_probs = self.policy_net(state_tensor).detach().numpy()[0]
        return np.random.choice([0, 1], p=action_probs)  # Sample action based on probabilities
    
    def learn(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            rewards = []
            log_probs = []
            done = False
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                
                # Store log probability of the action taken
                state_tensor = torch.tensor([[state]], dtype=torch.float32)
                action_prob = self.policy_net(state_tensor)[0][action]
                log_probs.append(torch.log(action_prob))
                
                rewards.append(reward)
                state = next_state
            
            # Compute returns (discounted future rewards)
            returns = []
            G_t = 0
            for r in reversed(rewards):
                G_t = r + (0.99 * G_t)  # Discount factor of 0.99
                returns.insert(0, G_t)
            
            # Normalize returns for stability
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
            
            # Update policy network using REINFORCE algorithm
            loss = -sum(log_prob * ret for log_prob, ret in zip(log_probs, returns))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

if __name__ == "__main__":
    env = SimpleGridEnv()
    agent = PolicyGradientAgent(env)

    num_episodes = 1000
    agent.learn(num_episodes)

    print("Training complete.")
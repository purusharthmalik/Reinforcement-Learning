import numpy as np
import random

class MultiArmedBandit:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
    
    def select_arm(self):
        if random.random() > self.epsilon:
            return np.argmax(self.values)
        else:
            return random.randint(0, self.n_arms - 1)
    
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value

true_ctrs = [0.05, 0.1, 0.02, 0.3]

def simulate(bandit, n_simulations):
    total_reward = 0
    chosen_arms = []
    
    for i in range(n_simulations):
        chosen_arm = bandit.select_arm()
        chosen_arms.append(chosen_arm)
        
        reward = np.random.binomial(1, true_ctrs[chosen_arm])
        
        bandit.update(chosen_arm, reward)
        total_reward += reward
    
    return total_reward, chosen_arms

n_arms = len(true_ctrs)
bandit = MultiArmedBandit(n_arms, epsilon=0.1)

n_simulations = 10000
total_reward, chosen_arms = simulate(bandit, n_simulations)

print(f"Total clicks after {n_simulations} impressions: {total_reward}")
print("Estimated CTRs for each ad slot:")
for i in range(n_arms):
    print(f"Slot {i} - Estimated CTR: {bandit.values[i]:.4f}, Selected {bandit.counts[i]} times")
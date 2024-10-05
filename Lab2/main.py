import numpy as np
import random

class NonstationaryMultiArmedBandit:
    def __init__(self, n_arms, epsilon=0.1, alpha=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.alpha = alpha
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
    
    def select_arm(self):
        if random.random() > self.epsilon:
            return np.argmax(self.values)
        else:
            return random.randint(0, self.n_arms - 1)
    
    def update(self, chosen_arm, reward):
        old_value = self.values[chosen_arm]
        new_value = (1 - self.alpha) * old_value + self.alpha * reward
        self.values[chosen_arm] = new_value

def generate_nonstationary_ctr(t, true_ctrs, change_points):
    for change_point in change_points:
        if t == change_point:
            true_ctrs = [max(0.0, min(1.0, ctr + np.random.uniform(-0.1, 0.1))) for ctr in true_ctrs]
    
    return true_ctrs

def simulate(bandit, n_simulations, initial_ctrs, change_points):
    total_reward = 0
    chosen_arms = []
    true_ctrs = initial_ctrs.copy()
    
    for t in range(n_simulations):
        true_ctrs = generate_nonstationary_ctr(t, true_ctrs, change_points)
        
        chosen_arm = bandit.select_arm()
        chosen_arms.append(chosen_arm)
        
        reward = np.random.binomial(1, true_ctrs[chosen_arm])
        
        bandit.update(chosen_arm, reward)
        total_reward += reward
    
    return total_reward, chosen_arms, true_ctrs

initial_ctrs = [0.05, 0.1, 0.02, 0.3]
n_arms = len(initial_ctrs)
epsilon = 0.1
alpha = 0.1
change_points = [2000, 5000, 8000]

bandit = NonstationaryMultiArmedBandit(n_arms, epsilon=epsilon, alpha=alpha)

n_simulations = 10000
total_reward, chosen_arms, final_ctrs = simulate(bandit, n_simulations, initial_ctrs, change_points)

print(f"Total clicks after {n_simulations} impressions: {total_reward}")
print("Final estimated CTRs for each ad slot after nonstationary simulation:")
for i in range(n_arms):
    print(f"Slot {i} - Estimated CTR: {bandit.values[i]:.4f}, Selected {bandit.counts[i]} times")
print("Final true CTRs for each slot:")
for i, ctr in enumerate(final_ctrs):
    print(f"Slot {i} - True CTR: {ctr:.4f}")
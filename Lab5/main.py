import numpy as np
import random

class GridWorld:
    def __init__(self, size, start, goal):
        self.size = size
        self.start = start
        self.goal = goal
        self.state = start
        
    def reset(self):
        self.state = self.start
        return self.state
    
    def step(self, action):
        if action == 0:  # Up
            next_state = (max(self.state[0] - 1, 0), self.state[1])
        elif action == 1:  # Down
            next_state = (min(self.state[0] + 1, self.size[0] - 1), self.state[1])
        elif action == 2:  # Left
            next_state = (self.state[0], max(self.state[1] - 1, 0))
        elif action == 3:  # Right
            next_state = (self.state[0], min(self.state[1] + 1, self.size[1] - 1))
        
        reward = 1 if next_state == self.goal else 0
        done = next_state == self.goal
        
        self.state = next_state
        return next_state, reward, done
    
    def get_actions(self):
        return [0, 1, 2, 3]  # Up, Down, Left, Right

def monte_carlo_policy_evaluation(env, policy, num_episodes=1000, gamma=0.9):
    V = np.zeros(env.size)
    returns = {state: [] for state in np.ndindex(env.size)}
    
    for _ in range(num_episodes):
        state = env.reset()
        episode = []
        
        done = False
        while not done:
            action = policy[state]
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            state = next_state
        
        G = sum(gamma ** t * r for t, (_, _, r) in enumerate(episode))
        
        for state, _, _ in episode:
            returns[state].append(G)
            V[state] = np.mean(returns[state]) if returns[state] else V[state]
    
    return V

def monte_carlo_policy_improvement(env, V):
    policy = {}
    for state in np.ndindex(env.size):
        q_values = np.zeros(len(env.get_actions()))
        
        for action in env.get_actions():
            total_return = 0
            num_samples = 0
            
            # Sample episodes to estimate Q-values
            for _ in range(100):  # Sample size for estimating Q-values
                state_temp = env.reset()
                done = False
                
                while not done:
                    if state_temp == state:
                        action_temp = action
                    else:
                        action_temp = policy.get(state_temp, random.choice(env.get_actions()))
                    
                    next_state_temp, reward_temp, done_temp = env.step(action_temp)
                    
                    if done_temp:
                        total_return += reward_temp
                        num_samples += 1
                    
                    state_temp = next_state_temp
            
            q_values[action] = total_return / num_samples if num_samples > 0 else 0
        
        policy[state] = np.argmax(q_values)
    
    return policy

def monte_carlo_control(env, num_episodes=1000):
    policy = {state: random.choice(env.get_actions()) for state in np.ndindex(env.size)}
    
    for _ in range(num_episodes):
        V = monte_carlo_policy_evaluation(env, policy)
        policy = monte_carlo_policy_improvement(env, V)
    
    return policy, V

# Create GridWorld environment with size (4x4), starting at (0,0) and goal at (3,3)
env_size = (4, 4)
start_position = (0, 0)
goal_position = (3, 3)

grid_world_env = GridWorld(size=env_size, start=start_position, goal=goal_position)

# Run Monte Carlo Control to find the optimal policy and value function
optimal_policy, optimal_value_function = monte_carlo_control(grid_world_env)

print("Optimal Policy:")
print(optimal_policy)
print("Optimal Value Function:")
print(optimal_value_function)

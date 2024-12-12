import numpy as np

class MDP:
    def __init__(self, states, actions, transition_probs, rewards):
        self.states = states
        self.actions = actions
        self.transition_probs = transition_probs  # A dict of dicts: P[s][a] = {s': prob}
        self.rewards = rewards  # A dict: rewards[s] = reward for state s

def policy_evaluation(mdp, policy, gamma=0.9, tol=1e-6):
    V = np.zeros(len(mdp.states))
    while True:
        delta = 0
        for s in mdp.states:
            v = V[s]
            V[s] = sum(mdp.transition_probs[s][policy[s]].get(s_prime, 0) * 
                       (mdp.rewards[s] + gamma * V[s_prime]) 
                       for s_prime in mdp.states)
            delta = max(delta, abs(v - V[s]))
        if delta < tol:
            break
    return V

def policy_improvement(mdp, V, gamma=0.9):
    policy = {}
    for s in mdp.states:
        q_values = np.zeros(len(mdp.actions))
        for a in mdp.actions:
            q_values[a] = sum(mdp.transition_probs[s][a].get(s_prime, 0) * 
                              (mdp.rewards[s] + gamma * V[s_prime]) 
                              for s_prime in mdp.states)
        policy[s] = mdp.actions[np.argmax(q_values)]
    return policy

def policy_iteration(mdp, gamma=0.9):
    # Initialize a random policy
    policy = {s: np.random.choice(mdp.actions) for s in mdp.states}
    
    while True:
        V = policy_evaluation(mdp, policy, gamma)
        new_policy = policy_improvement(mdp, V, gamma)
        
        if new_policy == policy:  # Check for convergence
            break
        
        policy = new_policy
    
    return policy, V

# Define states and actions
states = [0, 1, 2]
actions = [0, 1]  # Assume action 0 and action 1

# Transition probabilities: P[state][action][next_state]
transition_probs = {
    0: {0: {0: 0.8, 1: 0.2}, 1: {0: 0.6, 2: 0.4}},
    1: {0: {1: 0.7, 2: 0.3}, 1: {1: 0.5, 2: 0.5}},
    2: {0: {2: 1.0},      1: {2: 1.0}}
}

# Rewards for each state
rewards = {0: -1, 1: -2, 2: 10}

# Create MDP instance
mdp_instance = MDP(states, actions, transition_probs, rewards)

# Run Policy Iteration
optimal_policy, optimal_value_function = policy_iteration(mdp_instance)

print("Optimal Policy:", optimal_policy)
print("Optimal Value Function:", optimal_value_function)
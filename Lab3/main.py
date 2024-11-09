import numpy as np

class MDP:
    def __init__(self, states, actions, transition_prob, rewards, gamma=0.9):
        self.states = states
        self.actions = actions
        self.transition_prob = transition_prob
        self.rewards = rewards
        self.gamma = gamma
        self.values = {s: 0.0 for s in states}
        self.policy = {s: np.random.choice(actions) for s in states}

    def value_iteration(self, theta=0.0001):
        iteration = 0
        while True:
            delta = 0
            new_values = self.values.copy()
            
            for s in self.states:
                action_values = []
                for a in self.actions:
                    expected_value = 0
                    for next_s in self.states:
                        prob = self.transition_prob.get((s, a, next_s), 0)
                        reward = self.rewards.get((s, a), 0)
                        expected_value += prob * (reward + self.gamma * self.values[next_s])
                    action_values.append(expected_value)
                
                best_action_value = max(action_values)
                new_values[s] = best_action_value
                delta = max(delta, abs(self.values[s] - new_values[s]))
                
            self.values = new_values
            
            if delta < theta:
                break
            
            iteration += 1
        
        self.derive_policy()

    def derive_policy(self):
        for s in self.states:
            action_values = {}
            for a in self.actions:
                expected_value = 0
                for next_s in self.states:
                    prob = self.transition_prob.get((s, a, next_s), 0)
                    reward = self.rewards.get((s, a), 0)
                    expected_value += prob * (reward + self.gamma * self.values[next_s])
                action_values[a] = expected_value
            
            self.policy[s] = max(action_values, key=action_values.get)

    def print_policy_and_values(self):
        print("Optimal Value Function:")
        for s in self.states:
            print(f"State {s}: Value = {self.values[s]:.4f}")
        
        print("\nOptimal Policy:")
        for s in self.states:
            print(f"State {s}: Best Action = {self.policy[s]}")

states = ["S1", "S2", "S3", "S4"]
actions = ["A1", "A2"]

transition_prob = {
    ("S1", "A1", "S2"): 0.5, ("S1", "A1", "S3"): 0.5,
    ("S2", "A2", "S3"): 0.7, ("S2", "A2", "S4"): 0.3,
    ("S3", "A1", "S1"): 0.4, ("S3", "A1", "S4"): 0.6,
    ("S4", "A2", "S1"): 1.0
}

rewards = {
    ("S1", "A1"): 5, ("S1", "A2"): 10,
    ("S2", "A1"): -1, ("S2", "A2"): 2,
    ("S3", "A1"): 7, ("S3", "A2"): -2,
    ("S4", "A1"): 3, ("S4", "A2"): 8
}

mdp = MDP(states, actions, transition_prob, rewards, gamma=0.9)
mdp.value_iteration()
mdp.print_policy_and_values()

import numpy as np
import matplotlib.pyplot as plt
import random


class Bandit:
    def __init__(self,  k_arm, epsilon, initial_estimates) -> None:
        self.initial_estimates = initial_estimates
        self.k_arm = k_arm
        self.epsilon = epsilon
        self.action_list = list(range(k_arm))  # q*(a), for a = 0, 1, ..., 9
        self.true_reward = 0.0

    def reset(self):
        # q*(a) is an normal distribution with mean 0 and variance 1 for each action a = 0, 1, ..., 9 [Figure 2.1]
        self.q_true = np.random.normal(loc=0.0, scale=1.0, size=self.k_arm)
        self.optimal_action = np.argmax(self.q_true)

        #  initial estimate Q1(a) = 0, for all a
        self.q_estimated = self.initial_estimates*1.0  # Q(A) = 0

        # counts the occurence in an action
        self.action_count = np.zeros(self.k_arm, dtype=int)  # N(A) = 0

    def action(self):
        # random variable epsilon-gredy simulation
        u = random.uniform(0, 1)
        if u < 1 - self.epsilon:
            return np.argmax(self.q_estimated)
        else:
            # select randomly an action list
            return random.choice(self.action_list)

    def step(self, action: int):
        # R_t has distribution normal with mean q*(A_t) and variance 1 [Figure 2.1]
        reward = random.gauss(self.q_true[action], 1)

        # update the occurrences in an action
        self.action_count[action] += 1

        # sample-average technique Q(A) = Q(A) + 1/N(A)*(R - Q(A))
        self.q_estimated[action] += (reward - self.q_estimated[action]) / self.action_count[action]

        return reward


def bandit_algorithm(runs, time: int, bandit: Bandit, T_MAX=1000):
    rewards = np.zeros((runs, time))
    optimal_actions = np.zeros((runs, time))

    # number of simulations to obtain the mean
    for run in range(runs):
        bandit.reset()
        # we collect the reward in t=0,1,...,999
        t = 0
        while t < T_MAX:
            action = bandit.action()
            reward = bandit.step(action)

            rewards[run, t] = reward

            if action == bandit.optimal_action:
                optimal_actions[run, t] += 1

            t += 1

    average_reward = rewards.mean(axis=0)
    average_optimal_actions = optimal_actions.mean(axis=0)
    return average_reward, average_optimal_actions


epsilon_values = [0.0, 0.01, 0.1]

runs = 2000
time = 1000
k_arm = 10
initial_estimates = np.zeros(k_arm)

fig, axes = plt.subplots(2, 1)

plt.suptitle("k-armed bandit")

axes[0].set_xlabel("steps")
axes[0].set_ylabel("Average reward")

axes[1].set_xlabel("steps")
axes[1].set_ylabel("% Optimal action")

for epsilon in epsilon_values:
    bandit = Bandit(k_arm, epsilon, initial_estimates)
    rewards, optimal = bandit_algorithm(runs, time, bandit)
    
    axes[0].plot(rewards, label='$\epsilon = %.02f$' % (epsilon))
    axes[1].plot(optimal, label='$\epsilon = %.02f$' % (epsilon))

axes[0].legend()
axes[1].legend()

plt.show()

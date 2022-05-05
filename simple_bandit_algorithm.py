from matplotlib.pyplot import axis
import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, k_arm=10, epsilon=0., initial_value=0) -> None:
        self.k_arm = k_arm
        self.epsilon = epsilon
        self.initial_value = initial_value
        self.time = 0
        self.bandits = np.arange(k_arm)  # [0, 1, ..., 9]
        self.true_reward = 0.0

    def reset(self):
        # each q is normal distributed and initial
        self.q_true = np.random.randn(self.k_arm) + self.true_reward

        # initially we don't now the true value of each q
        self.q_estimation = np.zeros(self.k_arm) + self.initial_value

        # counts the occurence in an action
        self.action_count = np.zeros(self.k_arm, dtype=int)

        self.time = 0

    def action(self):
        # random variable epsilon-gredy simulation
        u = np.random.uniform()
        if u < 1 - self.epsilon:
            return np.argmax(self.q_estimation)
        else:
            # select randomly an arm
            return np.random.choice(self.bandits)

    def step(self, action: int):
        # R_t has distribution normal with mean q*(A_t)
        reward = np.random.rand() + self.q_true[action]
        self.time += 1
        # update the occurrences in an action
        self.action_count[action] += 1

        # sample averages
        self.q_estimation[action] += (reward - self.q_estimation[action]
                                      )/self.action_count[action]

        return reward


def simulation(runs, time: int, bandit: Bandit):
    rewards = np.zeros((runs, time))
    # number of simulations to obtain the mean
    for run in range(runs):
        bandit.reset()
        # we collect the reward in t=0,1,...,999
        for t in range(time):
            action = bandit.action()
            reward = bandit.step(action)

            rewards[run, t] = reward
    mean_rewards = rewards.mean(axis=0)
    return mean_rewards

epsilon_values = [0, 0.01, 0.1]
    
runs = 2000
time = 1000
for epsilon in epsilon_values:
    bandit = Bandit(epsilon=epsilon)
    rewards = simulation(runs, time, bandit)
    plt.plot(rewards, label='$\epsilon = %.02f$' % (epsilon))

plt.xlabel('steps')
plt.ylabel('average reward')
plt.legend()
plt.show()
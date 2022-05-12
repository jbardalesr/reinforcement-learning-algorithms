import matplotlib.pyplot as plt
import random
import numpy as np


def bandit_algorithm(k_arm, epsilon, runs, time):
    collect_reward = np.zeros((runs, time))
    collect_op_action = np.zeros((runs, time))

    action_list = list(range(k_arm))

    for run in range(runs):
        # Initialize, for a = 1 to k
        # q*(a) is an normal distribution with mean 0 and variance 1 for each action a = 0, 1, ..., 9 [Figure 2.1]
        q_true = np.random.normal(0, 1, k_arm)  
        optimal_action = np.argmax(q_true)

        # initial estimate Q1(a) = 0, for all a
        q_estimated = np.zeros(k_arm)

        # counts the occurence in an action    
        action_count = [0]*k_arm            

        # Loop forever
        for t in range(time):
            # random variable epsilon-gredy simulation
            if random.uniform(0, 1) < 1 - epsilon:
                action = np.argmax(q_estimated)
            else:
                action = random.choice(action_list)

            # R_t has distribution normal with mean q*(A_t) and variance 1 [Figure 2.1]
            R = random.gauss(q_true[action], 1)
            # update the occurrences in an action
            action_count[action] = action_count[action] + 1
            # sample-average technique Q(A) = Q(A) + 1/N(A)*(R - Q(A))
            q_estimated[action] = q_estimated[action] + (R - q_estimated[action])/action_count[action]

            # collect the reward tp print
            collect_reward[run, t] = R

            if action == optimal_action:
                collect_op_action[run, t] += 1

    return collect_reward.mean(axis=0), collect_op_action.mean(axis=0)


epsilon_values = [0.0, 0.01, 0.1]

runs = 2000
time = 1000
k_arm = 10

fig, axes = plt.subplots(2, 1)
plt.suptitle("k-armed bandit")

axes[0].set_xlabel("steps")
axes[0].set_ylabel("Average reward")

axes[1].set_xlabel("steps")
axes[1].set_ylabel("% Optimal action")

for epsilon in epsilon_values:
    rewards, optimal = bandit_algorithm(k_arm, epsilon, runs, time)

    axes[0].plot(rewards, label='$\epsilon = %g$' % (epsilon))
    axes[1].plot(optimal, label='$\epsilon = %g$' % (epsilon))

axes[0].legend()
axes[1].legend()

plt.show()

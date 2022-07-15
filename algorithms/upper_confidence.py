from gc import collect
import matplotlib.pyplot as plt
import random
import numpy as np
import math


def upper_confidence(k_arm: int, runs: int, time: int, c: float):
    collect_reward = np.zeros((runs, time))
    for run in range(runs):
        # Initialize, for a = 1 to k
        # q*(a) is an normal distribution with mean 0 and variance 1 for each action a = 0, 1, ..., 9 [Figure 2.1]
        q_true = np.random.normal(0, 1, k_arm)
        # initial estimate Q1(a) = 0, for all a
        q_estimated = np.zeros(k_arm)
        # count the occurrene in an action
        action_count = np.zeros(k_arm)
        # Loop forever
        for t in range(time):
            action = np.argmax(q_estimated + c * np.sqrt(math.log(t + 1) / action_count))
            # R_t has distribution normal with mean q*(A_t) and variance 1 [Figure 2.1]
            reward = random.gauss(q_true[action], 1)
            action_count[action] += 1
            # sample-average technique Q(A) = Q(A) + 1/N(A)*(reward - Q(A))
            q_estimated[action] += (reward - q_estimated[action]) / action_count[action]
            # collect the reward to print
            collect_reward[run, t] = reward

    return collect_reward.mean(axis=0)

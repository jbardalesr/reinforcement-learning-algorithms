
import numpy as np
import gym

# first, we create the envinroment, initialization the value of policy
env = gym.make("FrozenLake-v1")
env = env.unwrapped

nA = env.action_space.n         # 0 1 2 3
nS = env.observation_space.n    # all state that we can see

# initialize with zeros
V = np.zeros(nS)
policy = np.zeros(nS)


def eval_state_action(V: np.ndarray, s, a, gamma=0.99):
    """
    gamma: is the discount factor
    """
    # env.P is a dictionary thta contains all the information about the dynamic of the envinroment
    return np.sum([p * (rew + gamma * V[next_s]) for p, next_s, rew, _ in env.P[s][a]])


def policy_evaluation(V: np.ndarray, policy, eps=0.0001):
    """
    Policy evaluation. Update the value function until it reach a steady state
    Because the policy is deterministic, we only evaluate one action.
    """
    while True:
        delta = 0
        for s in range(nS):
            old_v = V[s]
            V[s] = eval_state_action(V, s, policy[s])
            delta = max(delta, np.abs(old_v - V[s]))
        # we consider the value function stable whenever delta is lower than threshold, eps
        if delta < eps:
            break


def policy_improvement(V, policy):
    '''
    Policy improvement. Update the policy based on the value function
    '''
    policy_stable = True
    for s in range(nS):
        old_a = policy[s]
        # update the policy with the action that bring to the highest state value
        policy[s] = np.argmax([eval_state_action(V, s, a) for a in range(nA)])
        if old_a != policy[s]:
            policy_stable = False

    return policy_stable


def run_episodes(env: gym.Env, policy, num_games=100):
    '''
    Run some games to test a policy
    '''
    tot_rew = 0
    state = env.reset()
    for _ in range(num_games):
        done = False
        while not done:
            next_state, reward, done, _ = env.step(policy[state])
            state = next_state
            tot_rew += reward
            if done:
                state = env.reset()
    print('Won %i of %i games!' % (tot_rew, num_games))


if __name__ == '__main__':
    # create the environment
    env = gym.make('FrozenLake-v1')
    # enwrap it to have additional information from it
    env = env.unwrapped

    # spaces dimension
    nA = env.action_space.n
    nS = env.observation_space.n

    # initializing value function and policy
    V = np.zeros(nS)
    policy = np.zeros(nS)

    # some useful variable
    policy_stable = False
    it = 0

    while not policy_stable:
        policy_evaluation(V, policy)
        policy_stable = policy_improvement(V, policy)
        it += 1

    print('Converged after %i policy iterations' % (it))
    run_episodes(env, policy)
    print(V.reshape((4, 4)))
    print(policy.reshape((4, 4)))

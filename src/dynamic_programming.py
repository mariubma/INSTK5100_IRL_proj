import numpy as np


def state_action_value(s, a, p, rewards, gamma, V):
    n_states = np.shape(V)[0]
    Q = 0

    for next_state in range(n_states):
        for r in range(len(rewards)):
            Q += p[next_state, r, s, a] * (rewards[r] + gamma * V[next_state])
    return Q


def value_iteration(
    theta,
    gamma,
    rewards,
    p,
    n_actions,
    n_states,
):
    V = np.zeros(n_states)
    pi = np.ones((n_states, n_actions)) / n_actions
    cont = True
    while cont:
        delta = 0
        for s in range(n_states):
            v = V[s]
            Q = [
                state_action_value(s, a, p, rewards, gamma, V) for a in range(n_actions)
            ]
            V[s] = max(Q)
            delta = max([delta, abs(v - V[s])])
        if delta < theta:
            cont = False
    for s in range(n_states):
        Q = [state_action_value(s, a, p, rewards, gamma, V) for a in range(n_actions)]
        opt_action_ind = np.argmax(Q)
        pi[s] = np.eye(n_actions)[opt_action_ind]
    return V, pi


def print_policy(pi, size=5):
    n_states = np.shape(pi)[0]
    actions = np.argmax(pi, axis=1)
    unicodes = []
    for s in range(n_states):
        if actions[s] == 0:
            unicodes.append("\u2193")
        if actions[s] == 1:
            unicodes.append("\u2192")
        if actions[s] == 2:
            unicodes.append("\u2191")
        if actions[s] == 3:
            unicodes.append("\u2190")
    print("+" + "---+" * size)
    for s in range(0, n_states, size):
        print(("|" + " {} |" * size).format(*[unicodes[x] for x in range(s, s + size)]))
        print("+" + "---+" * size)

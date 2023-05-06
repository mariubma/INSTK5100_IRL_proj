import numpy as np
from scipy.optimize import linprog


def get_policy_action(policy, state):
    """
    Retrieves the index of the action prescribed by the policy in the given state.

    Args:
        policy (numpy.ndarray): The policy.
        state (int): The state index.

    Returns:
        int: The action index.
    """
    action_probs = policy[state, :]
    policy_action = np.argmax(action_probs)
    return policy_action


def get_occupancy_matrix(policy, transition_probs, gamma):
    """
    Computes the occupancy matrix, which is used in the computation of the
    objective function and the inequality constraints for linear programming.

    Args:
        policy (numpy.ndarray):
            The expert policy.
        transition_probs (numpy.ndarray):
            The three-argument transition probability matrix,
                        p(s'|s,a),
            as defined on page 49 of Sutton & Barto 2018.
        gamma (float): The discount factor for rewards.

    Returns:
        numpy.ndarray: The occupancy matrix.
    """
    n_states = policy.shape[0]

    expert_transition_probs = np.zeros((n_states, n_states))

    for s in range(n_states):
        expert_action = get_policy_action(policy, s)
        row_s = transition_probs[:, s, expert_action]
        expert_transition_probs[s, :] = row_s

    occ_mat = np.linalg.inv(np.identity(n_states) - gamma * expert_transition_probs)
    return occ_mat


def get_constraint_matrix(policy, transition_probs, gamma):
    """
    Computes the linear inequality constraint matrix for linear programming.
    See Sections 3.1 and 3.2 of Ng & Russell, 2000 for further details.

    Args:
        policy (numpy.ndarray):
            The expert policy.
        transition_probs (numpy.ndarray):
            The three-argument transition probability matrix,
                        p(s'|s,a),
            as defined on page 49 of Sutton & Barto 2018.
        gamma (float): The discount factor for rewards.

    Returns:
        numpy.ndarray: the constraint matrix.
    """
    n_states = policy.shape[0]
    n_actions = policy.shape[1]

    occ_mat = get_occupancy_matrix(
        policy,
        transition_probs,
        gamma,
    )
    constr_mat = np.zeros((n_states * (n_actions - 1), n_states))

    for s in range(n_states):
        expert_action = get_policy_action(policy, s)

        non_expert_actions = [a for a in range(n_actions) if a != expert_action]
        intra_stateblock_ind = 0
        for a in non_expert_actions:
            diff_term = (
                transition_probs[:, s, expert_action] - transition_probs[:, s, a]
            )
            state_action_constr = np.matmul(diff_term.T, occ_mat)

            stateblock_ind = s * (n_actions - 1)
            constr_mat[stateblock_ind + intra_stateblock_ind, :] = state_action_constr
            intra_stateblock_ind += 1
    return constr_mat


def get_ng_russell2000_objectve_coefficients(policy, transition_probs, gamma, _lambda):
    """
    Computes the coefficients of the objective function
    as detailed in Section 3.2 of Ng & Russell, 2000.

    Args:
        policy (numpy.ndarray):
            The expert policy.
        transition_probs (numpy.ndarray):
            The three-argument transition probability matrix,
                        p(s'|s,a),
            as defined on page 49 of Sutton & Barto 2018.
        gamma (float): The discount factor for rewards.
        _lambda (float): The penalty coefficient.

    Returns:
        numpy.ndarray: The coefficients for the objective function.
    """
    n_states = policy.shape[0]
    n_actions = policy.shape[1]

    occ_mat = get_occupancy_matrix(
        policy,
        transition_probs,
        gamma,
    )
    obj_coeffs = np.zeros(n_states)

    for s in range(n_states):
        expert_action = get_policy_action(policy, s)

        diff_term = np.ones(n_states) * np.inf
        non_expert_actions = [a for a in range(n_actions) if a != expert_action]
        for a in non_expert_actions:
            action_diff = (
                transition_probs[:, s, expert_action] - transition_probs[:, s, a]
            )
            if np.linalg.norm(diff_term) > np.linalg.norm(action_diff):  # correct?
                diff_term = action_diff
            state_constr = np.matmul(diff_term.T, occ_mat)
            obj_coeffs += state_constr
    return obj_coeffs - np.ones(n_states) * _lambda


def infer_reward(transition_probs, policy, gamma, _lambda):
    """
    Infers the reward function from a given policy and transition dynamics
    using linear programming (LP), as detailed in Section 3.2 of Ng & Russell, 2000.

    NB:
        scipy.optimize.linprog is based on a formulation of LP in which the objective
        function is minimized, subject to inequality constraints of the form
        A_ub @ x <= b_ub. Ng & Russell 2000, on the other hand, formulates the LP
        problem as a maximization problem s.t. constraints of the form
        A_lb @ x >= b_lb. We therefore take the negative of the arrays involved to
        obtain a minimization formulation of the given LP problem.

    Args:
        transition_probs (numpy.ndarray):
            The four-argument transition probability matrix,
                        p(s',r|s,a),
            as defined on page 48 of Sutton & Barto 2018.
        policy (numpy.ndarray):
            The expert policy.
        gamma (float): The discount factor for rewards.
        _lambda (float): The penalty coefficient.

    Returns:
        numpy.ndarray: The estimated reward function.
    """
    n_states = policy.shape[0]
    size = int(np.sqrt(n_states))
    P = transition_probs.sum(axis=1)
    A_lb = get_constraint_matrix(policy, P, gamma)
    n_constrs = A_lb.shape[0]
    b_lb = np.zeros(n_constrs)
    c_max = get_ng_russell2000_objectve_coefficients(policy, P, gamma, _lambda)
    c_min, A_ub, b_ub = -c_max, -A_lb, -b_lb
    r_est = linprog(c_min.T, A_ub=A_ub, b_ub=b_ub, bounds=(0, 1)).x
    return np.reshape(r_est, (size, size))

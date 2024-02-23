import numpy as np

def create_transition_matrix(num_states: int, p_win: float)-> np.ndarray:
    """
    Create a transition matrix for a random walk with absorbing states at 0 and
    num_states
    """
    transition_matrix = np.zeros((num_states, num_states))
    # Fill in the transition matrix
    # For win and lose states
    # Uses vectorized assignment instead of for loop for efficiency
    start_idx = np.arange(1,num_states-1)
    transition_matrix[start_idx, start_idx - 1] = 1-p_win
    transition_matrix[start_idx, start_idx + 1] = p_win

    # Fill in the absorbing states
    transition_matrix[0, 0] = 1.0 # absorbing state for ruin
    transition_matrix[-1, -1] = 1.0 # absorbing state for success

    return transition_matrix

def find_nth_state(transition_matrix: np.ndarray,
                   initial_state: np.ndarray, n: int)-> np.ndarray:
    """
    Find the state after n steps
    """
    # @ operator works for matrix multiplication in numpy
    # no need to cast to np.matrix
    return np.linalg.matrix_power(transition_matrix, n) @ initial_state

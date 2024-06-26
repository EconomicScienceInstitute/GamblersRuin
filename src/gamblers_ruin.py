import numpy as np

def create_policy_function(num_states: int, p_win: float)-> np.ndarray:
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
    return initial_state @ np.linalg.matrix_power(transition_matrix, n)


def find_expected_value(state_map: np.ndarray,
                        state: np.ndarray)-> float:
    """
    Finds the expected value of the cash on hand for the gamblers ruin
    problem

    Parameters
    ----------
    state_map
        List of values for each state with corresponding index
    state
        List of probabilities for each state with corresponding index

    Returns
    -------
        float value for expected value of the current state
    """
    # The cross product of the state_map and state gives the expected value
    expected_value = np.sum(state_map * state)
    return expected_value

def create_lose_states(start_cash: int,
                       min_bet: int)-> np.ndarray:
    if start_cash <= 0 or min_bet <= 0:
        raise ValueError("start_cash and min_bet must be greater than 0")
    cash = start_cash
    count = 0
    lose_states = []
    while cash > 0:
        cash -= min_bet * 2**count
        count += 1
        lose_states.append(cash)
    lose_states[-1] = 0
    return np.flip(np.array(lose_states))

def create_state_map(start_cash: int,
                         min_bet: int,
                         goal: int)-> np.ndarray:
    """
    Create the initial state for the gambler's ruin problem
    """
    win_states = np.arange(start_cash, goal+min_bet, min_bet)
    lose_states = create_lose_states(start_cash, min_bet)
    start_idx = lose_states.size
    return np.append(lose_states, win_states), start_idx

def run_gamblers_ruin(start_cash: int,
                        min_bet: int,
                        goal: int,
                        p: float,
                        period: int)->np.ndarray:
    if not 0 < p <= 1:
        raise ValueError("p (probability of winning) must be between 0 and 1")
    # Create the initial state
    state_map, start_idx = create_state_map(start_cash, min_bet, goal)

    # create the state_map
    initial_state = np.zeros(state_map.size)
    initial_state[start_idx] = 1.0

    # Create the transition matrix
    transition_matrix = create_policy_function(state_map.size, p)

    # Find the expected value of the current state
    current_state = find_nth_state(transition_matrix, initial_state, period)
    return current_state

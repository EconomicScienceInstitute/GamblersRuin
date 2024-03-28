import numpy as np


# Import functions from the 'gamblers_ruin' module
from gamblers_ruin import (create_transition_matrix, find_nth_state,
                           find_expected_value, create_lose_states,
                           create_state_map, run_gamblers_ruin)


def test_create_transition_matrix():
    """
    Test function for creating the transition matrix.

    This function tests the creation of a transition matrix
    based on the provided state map and probability of winning.
    It checks the shape of the matrix and verifies that the
    absorbing states are correctly set.
    """
    # Define the state map representing the possible states of the system
    state_map = np.append(np.array([0, 150, 300, 450]), np.arange(500, 1050, 50))
    # Probability of winning a single bet
    p_win = 17 / 36
    # Create the transition matrix
    transition_matrix = create_transition_matrix(state_map.shape[0], p_win)
    # Check the shape of the transition matrix
    assert transition_matrix.shape == (state_map.shape[0], state_map.shape[0])
    
    # Check the absorbing states (starting state and losing all money state)
    assert np.all(
        transition_matrix[0] == np.array([1.0] + [0.0] * (state_map.shape[0] - 1))
    )
    assert np.all(
        transition_matrix[-1] == np.array([0.0] * (state_map.shape[0] - 1) + [1.0])
    )


def test_find_nth_state():
    """
    Test function for finding the state after n steps.

    This function tests the behavior of the 'find_nth_state' function
    by simulating the transition of the system through multiple steps.
    It verifies that the probabilities are correctly distributed
    across different states after each step and ensures that the total
    probability sums up to 1.
    """
    # Define the state map representing the possible states of the system
    state_map = np.append(np.array([0, 150, 300, 450]), np.arange(500, 1050, 50))
    # Set the initial state (with all probability concentrated at a specific state)
    initial_state = np.zeros([state_map.shape[0],1])
    initial_state[:,4] = 1  # All probability initially at state 500
    initial_state = np.transpose(initial_state)
    # Probability of winning a single bet
    p_win = 17 / 36
    # Create the transition matrix
    transition_matrix = create_transition_matrix(state_map.shape[0], p_win)
    
    # Check the state after 0 steps
    state_0 = find_nth_state(transition_matrix, initial_state, 0)
    assert np.all(state_0[6:,:] == 0)  # Check absorbing states
    assert np.all(state_0[:4,:] == 0)   # Check other states
    assert state_0[4] == 1              # Check initial state
    
    # Check the state after 1 step
    state_1 = find_nth_state(transition_matrix, initial_state, 1)
    assert np.all(state_1[6:] == 0)     # Check absorbing states
    assert np.all(state_1[:3] == 0)     # Check other states
    assert state_1[3] == 1 - p_win      # Probability of being at state 300
    assert state_1[4] == 0              # Probability of being at state 500
    assert state_1[5] == p_win          # Probability of being at state 650
    
    # Check that the law of total probability holds for subsequent steps
    for i in range(2, 10):
        state = find_nth_state(transition_matrix, state_1, i)
        assert np.allclose(np.sum(state), 1, atol=1e-5)  # Ensure total probability sums to 1

import numpy as np
import pytest

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

def test_create_policy_function():
    """
    Test the create_policy_function function.

    This function tests whether the created policy matrix meets several criteria:
    1. Correct dimensions.
    2. Probabilities are valid (between 0 and 1).
    3. Sum of probabilities in each row is equal to 1.
    4. Absorbing states are correctly identified.
    5. Transition probabilities for non-absorbing states are correct.

    These tests ensure that the policy function generates a valid transition matrix,
    this is needed for gambler's ruin.
    """
    # Example parameters
    num_states = 5  # Number of states in the matrix
    p_win = 17/36     # Probability of moving to the next higher state

    # Call the function to test
    matrix = create_policy_function(num_states, p_win)

    # 1. Check the dimensions of the matrix
    assert matrix.shape == (num_states, num_states), "Matrix dimensions are incorrect."

    # 2. Verify the probabilities
    assert np.all(matrix >= 0) and np.all(matrix <= 1), "Matrix contains invalid probability values."


    # 3. Check the sum of probabilities for each row
    for row_sum in np.sum(matrix, axis=1):
        assert row_sum == 1, "The sum of probabilities in a row is not equal to 1."

    # 4. Verify the absorbing states
    assert matrix[0, 0] == 1 and np.all(matrix[0, 1:] == 0), "The first row is not an absorbing state."
    assert matrix[-1, -1] == 1 and np.all(matrix[-1, :-1] == 0), "The last row is not an absorbing state."

    # 5. Check the transition probabilities for non-absorbing states
    for i in range(1, num_states - 1):
        assert matrix[i, i - 1] == 1 - p_win, f"Incorrect transition probability at {i}, {i-1}"
        assert matrix[i, i + 1] == p_win, f"Incorrect transition probability at {i}, {i+1}"

    print("All tests passed.")

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

    Test the find_nth_state function.

    This function checks whether the find_nth_state function correctly computes the state
    after a certain number of transitions based on the transition matrix.

    These tests ensure that the function accurately calculates the next state given the transition matrix
    and the current state.
    """
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

        assert np.allclose(np.sum(state), 1, atol=1e-5)

def test_find_expected_value():
    """
    Test the find_expected_value function.

    This function verifies whether the find_expected_value function correctly computes
    the expected value based on the state map and current state.

    This test ensures that the function accurately calculates the expected value of the gambler's cash
    based on the current distribution of probabilities across different cash states.
    """
    state_map=np.array([0,1,1,0])
    current_state=np.array([10,20,30,40])
    assert find_expected_value(state_map,current_state)==50

    # second set of tests?
    state_map = np.array([0, 150, 350, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000])
    state = np.zeros(15)
    state[4] = 1  # All probability on starting cash
    expected_value = find_expected_value(state_map, state)
    assert expected_value == 500  # Expected value should be the starting cash

def test_create_state_map():
    """
    Test the create_state_map function.

    This function checks whether the create_state_map function correctly generates the state map
    and identifies the starting index based on the specified parameters.

    These tests ensure that the state map is created correctly, this helps represent
    the possible cash states of the gambler and identifying the starting cash index.
    """
    states= np.array([0,150,350,450,500,550,600,650,700,750,800,850,900,950,1000])
    test_state,start_index= create_state_map(500,50,1000)
    assert np.all(states == test_state)
    assert start_index==4
    states=np.array([0,1,2])
    test_state,start_index=create_state_map(1,1,2)
    assert np.all(states == test_state)
    assert start_index==1

    # second set of tests?
    start_cash = 500
    min_bet = 50
    goal = 1000
    expected_state_map = np.array([0, 150, 350, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000])
    state_map, start_idx = create_state_map(start_cash, min_bet, goal)
    assert np.array_equal(state_map, expected_state_map)
    assert start_idx == 4  # Index where the starting cash is located

    # test for boundary condition
    state_map, _ = create_state_map(50, 50, 100)
    assert len(state_map) > 0  # Ensure state map is created correctly

def test_create_lose_states():
    """
    Test the create_lose_states function.

    This function checks whether the create_lose_states function correctly generates
    losing states based on the specified parameters.

    This test ensures that the function properly identifies the losing states within the state map,
    this helps with simulating the termination condition of the gambler's ruin problem.
    """
    expected_lose_states = np.array([0, 150, 350, 450])
    assert np.array_equal(create_lose_states(500, 50), expected_lose_states)

    # test for negative input
    with pytest.raises(ValueError):
        create_lose_states(-500, 50)

def test_run_gamblers_ruin():
    """
    Test the run_gamblers_ruin function.

    This function tests whether the run_gamblers_ruin function correctly simulates
    the gambler's ruin problem given the input parameters.

    These tests  are donw to verify that the function accurately simulates the progression of the gambler's cash
    over time and accounts for different scenarios such as zero probability of winning or large numbers.
    """
    start_cash = 500
    min_bet = 50
    goal = 1000
    p_win = 0.5
    period = 1
    current_state = run_gamblers_ruin(start_cash, min_bet, goal, p_win, period)
    # Assert the shape of the current_state to ensure it's calculated
    assert current_state.shape[0] == 15  # Based on the state map size

    # test when zero probability
    with pytest.raises(ValueError):
        run_gamblers_ruin(500, 50, 1000, 0, 1)

    # test for boundary goal
    current_state = run_gamblers_ruin(500, 50, 550, 0.5, 1)
    assert current_state is not None  # Ensure it returns a valid state

    # test with large numbers
    current_state = run_gamblers_ruin(1000000, 50000, 2000000, 0.5, 1)
    assert current_state is not None  # Check for successful execution
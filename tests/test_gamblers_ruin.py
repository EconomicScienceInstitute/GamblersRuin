import numpy as np
import pytest

from gamblers_ruin import (create_policy_function, find_nth_state,
                           find_expected_value, create_lose_states,
                           create_state_map, run_gamblers_ruin)


def test_create_policy_function():
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
    """_summary_"""
    state_map = np.append(np.array([0, 150, 300, 450]), np.arange(500, 1050, 50))
    initial_state = np.zeros([state_map.shape[0],1])
    initial_state[4,:] = 1
    initial_state = np.transpose(initial_state)
    initial_state[:,4] = 1
    
    p_win = 17 / 36
    transition_matrix = create_policy_function(state_map.shape[0], p_win)
    # Check the state for initial
    state_0 = find_nth_state(transition_matrix, initial_state, 0)
    assert np.all(state_0[:,5:] == 0)
    assert np.all(state_0[:,:4] == 0)
    assert state_0[0,4] == 1

    # Check the state after 1 step
    state_1 = find_nth_state(transition_matrix, initial_state, 1)
    assert np.all(state_1[:,6:] == 0)
    assert np.all(state_1[:,:3] == 0)
    assert state_1[0,3] == 1 - p_win
    assert state_1[0,4] == 0
    assert state_1[0,5] == p_win

    # check that the law of total probability holds
    for i in range(2, 10):
        state = find_nth_state(transition_matrix, state_1, i)
        assert np.allclose(np.sum(state), 1, atol=1e-5)
def test_expected_value():
    state_map=np.array([0,1,1,0])
    current_state=np.array([10,20,30,40])
    assert find_expected_value(state_map,current_state)==50
def test_create_state_map():
    states= np.array([0,150,350,450,500,550,600,650,700,750,800,850,900,950,1000])
    test_state,start_index= create_state_map(500,50,1000)
    assert np.all(states == test_state)
    assert start_index==4
    states=np.array([0,1,2])
    test_state,start_index=create_state_map(1,1,2)
    assert np.all(states == test_state)
    assert start_index==1
def test_create_lose_states():
    expected_lose_states = np.array([0, 150, 350, 450])
    assert np.array_equal(create_lose_states(500, 50), expected_lose_states)


def test_create_state_map():
    start_cash = 500
    min_bet = 50
    goal = 1000
    expected_state_map = np.array([0, 150, 350, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000])
    state_map, start_idx = create_state_map(start_cash, min_bet, goal)
    assert np.array_equal(state_map, expected_state_map)
    assert start_idx == 4  # Index where the starting cash is located

def test_run_gamblers_ruin():
    start_cash = 500
    min_bet = 50
    goal = 1000
    p_win = 0.5
    period = 1
    current_state = run_gamblers_ruin(start_cash, min_bet, goal, p_win, period)
    # Assert the shape of the current_state to ensure it's calculated
    assert current_state.shape[0] == 15  # Based on the state map size

def test_find_expected_value():
    state_map = np.array([0, 150, 350, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000])
    state = np.zeros(15)
    state[4] = 1  # All probability on starting cash
    expected_value = find_expected_value(state_map, state)
    assert expected_value == 500  # Expected value should be the starting cash

def test_create_lose_states_negative_input():
    with pytest.raises(ValueError):
        create_lose_states(-500, 50)

def test_run_gamblers_ruin_zero_probability():
    with pytest.raises(ValueError):
        run_gamblers_ruin(500, 50, 1000, 0, 1)

def test_create_state_map_boundary_condition():
    state_map, _ = create_state_map(50, 50, 100)
    assert len(state_map) > 0  # Ensure state map is created correctly

def test_run_gamblers_ruin_boundary_goal():
    current_state = run_gamblers_ruin(500, 50, 550, 0.5, 1)
    assert current_state is not None  # Ensure it returns a valid state

def test_run_gamblers_ruin_large_numbers():
    current_state = run_gamblers_ruin(1000000, 50000, 2000000, 0.5, 1)
    assert current_state is not None  # Check for successful execution

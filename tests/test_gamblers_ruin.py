import numpy as np


from gamblers_ruin import (create_policy_function, find_nth_state,
                           find_expected_value, create_lose_states,
                           create_state_map, run_gamblers_ruin)


def test_create_transition_matrix():
    """_summary_"""
    state_map = np.append(np.array([0, 150, 300, 450]), np.arange(500, 1050, 50))
    p_win = 17 / 36
    transition_matrix = create_policy_function(state_map.shape[0], p_win)
    # Check the shape of the transition matrix
    assert transition_matrix.shape == (state_map.shape[0], state_map.shape[0])
    # Check the absorbing states
    assert np.all(
        transition_matrix[0] == np.array([1.0] + [0.0] * (state_map.shape[0] - 1))
    )
    assert np.all(
        transition_matrix[-1] == np.array([0.0] * (state_map.shape[0] - 1) + [1.0])
    )

def test_create_transition_matrix_edge_case():
    state_map = np.array([0, 50])  # Edge case with minimal states
    p_win = 0.5
    transition_matrix = create_transition_matrix(state_map.shape[0], p_win)
    expected_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])  # Absorbing states only
    assert np.array_equal(transition_matrix, expected_matrix)

def test_create_transition_matrix_non_absorbing():
    state_map = np.array([0, 50, 100, 150, 200, 250, 300])
    p_win = 0.5
    transition_matrix = create_transition_matrix(state_map.shape[0], p_win)
    # Test non-absorbing states transitions
    for i in range(1, state_map.shape[0] - 1):
        assert transition_matrix[i, i - 1] == 1 - p_win
        assert transition_matrix[i, i + 1] == p_win

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
    assert np.all(state_0[6:,:] == 0)
    assert np.all(state_0[:4,:] == 0)
    assert state_0[4] == 1

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

def test_find_expected_value():
    state_map = np.array([0, 50, 100, 150, 200])
    state = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
    expected_value = find_expected_value(state_map, state)
    assert expected_value == 110  # This should match the calculated expected value

def test_create_lose_states():
    start_cash = 500
    min_bet = 50
    expected_lose_states = np.array([500, 400, 300, 200, 100, 0])
    lose_states = create_lose_states(start_cash, min_bet)
    assert np.array_equal(lose_states, expected_lose_states)

def test_create_lose_states_edge_case():
    start_cash = 45  # Less than min_bet
    min_bet = 50
    expected_lose_states = np.array([0])  # Only possible lose state is bankruptcy
    lose_states = create_lose_states(start_cash, min_bet)
    assert np.array_equal(lose_states, expected_lose_states)

def test_create_state_map():
    start_cash = 500
    min_bet = 50
    goal = 1000
    expected_state_map = np.array([0, 100, 200, 300, 400, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000])
    state_map, start_idx = create_state_map(start_cash, min_bet, goal)
    assert np.array_equal(state_map, expected_state_map)
    assert start_idx == 5  # Index where the starting cash is located

def test_create_state_map_edge_case():
    start_cash = 100
    min_bet = 100
    goal = 200
    expected_state_map = np.array([0, 100, 200])
    state_map, start_idx = create_state_map(start_cash, min_bet, goal)
    assert np.array_equal(state_map, expected_state_map)
    assert start_idx == 1  # Index where the starting cash is located

def test_run_gamblers_ruin():
    start_cash = 500
    min_bet = 50
    goal = 1000
    p_win = 0.5
    period = 1
    current_state = run_gamblers_ruin(start_cash, min_bet, goal, p_win, period)
    # Test some basic properties of the current state

def test_run_gamblers_ruin_basic_scenario():
    start_cash = 100
    min_bet = 50
    goal = 200
    p_win = 0.5
    period = 1
    current_state = run_gamblers_ruin(start_cash, min_bet, goal, p_win, period)
    # Since this is a basic scenario with a fair game, check if the state probabilities are reasonable
    assert current_state[0] < 1  # Probability of ruin is less than 100%
    assert current_state[-1] < 1  # Probability of success is less than 100%

def test_run_gamblers_ruin_impossible_win():
    start_cash = 50
    min_bet = 100
    goal = 200
    p_win = 0.5
    period = 1
    current_state = run_gamblers_ruin(start_cash, min_bet, goal, p_win, period)
    # In this scenario, winning is impossible due to the bet size
    assert current_state[0] == 1  # Probability of ruin is 100%
    assert current_state[-1] == 0  # Probability of success is 0%
import numpy as np

from gamblers_ruin import create_transition_matrix, find_nth_state


def test_create_transition_matrix():
    """_summary_"""
    state_map = np.append(np.array([0, 150, 300, 450]), np.arange(500, 1050, 50))
    p_win = 17 / 36
    transition_matrix = create_transition_matrix(state_map.shape[0], p_win)
    # Check the shape of the transition matrix
    assert transition_matrix.shape == (state_map.shape[0], state_map.shape[0])
    # Check the absorbing states
    assert np.all(
        transition_matrix[0] == np.array([1.0] + [0.0] * (state_map.shape[0] - 1))
    )
    assert np.all(
        transition_matrix[-1] == np.array([0.0] * (state_map.shape[0] - 1) + [1.0])
    )


def test_find_nth_state():
    """_summary_"""
    state_map = np.append(np.array([0, 150, 300, 450]), np.arange(500, 1050, 50))
    initial_state = np.zeros([state_map.shape[0],1])
    initial_state[4,:] = 1
    initial_state = np.transpose(initial_state)
    p_win = 17 / 36
    transition_matrix = create_transition_matrix(state_map.shape[0], p_win)
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

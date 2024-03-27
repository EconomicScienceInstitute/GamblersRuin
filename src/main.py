import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from gamblers_ruin import (create_transition_matrix, find_nth_state, 
                           find_expected_value, run_gamblers_ruin)

# Set the title of the app 
st.title('Gambler\'s Ruin')
st.text('A simple app to simulate the gambler\'s ruin problem')

# Set the parameters for the random walk
starting_cash = st.slider('Starting Cash', 0, 1000, 500)
minimum_bet = st.slider('Minimum Bet', 0, 100, 50, step = 1)
goal_cash = st.slider('Goal Cash',
                      min_value=starting_cash,
                      max_value=starting_cash + 50*minimum_bet,
                      value=starting_cash + 10*minimum_bet,
                      step = minimum_bet)
p_win = st.slider('Probability of Winning', 0.0, 1.0, 17/36)
n_rounds = st.slider('Number of Rounds', 1, 1000, 100)
run_sim = st.button('Run Simulation')

def visualize_current_state(current_state: np.ndarray):
    """
    _summary_
    """
    # Create the plot
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(current_state)), current_state)
    ax.set_title('Current State of the Gambler\'s Ruin')
    ax.set_xlabel('Cash')
    ax.set_ylabel('Probability')
    st.pyplot(fig)

if run_sim:
    current_state = run_gamblers_ruin(starting_cash, minimum_bet, goal_cash, p_win, n_rounds)
    visualize_current_state(current_state)
    prob_ruin, prob_success = current_state[0], current_state[1]
    expected_value = find_expected_value(np.arange(0, starting_cash + 1, minimum_bet), current_state)
    st.write(f'The expected value of the current state is {expected_value}')
    st.write(f'The probability of ruin is {current_state[0]}')
    st.write(f'The probability of success is {current_state[-1]}')
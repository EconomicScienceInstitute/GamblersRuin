import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from gamblers_ruin import (create_transition_matrix, find_nth_state, 
                           find_expected_value, run_gamblers_ruin)

# Use columns for a cleaner layout
st.title('Gambler\'s Ruin')
st.markdown('A simple app to simulate the gambler\'s ruin problem')

# Using containers and columns for a better layout
with st.sidebar:
    st.header("Simulation Parameters")
    starting_cash = st.slider('Starting Cash', 0, 1000, 500, help="Initial amount of money the gambler starts with.")
    minimum_bet = st.slider('Minimum Bet', 1, 100, 50, help="The smallest amount that can be wagered.")
    goal_cash = st.slider('Goal Cash',
                          min_value=starting_cash,
                          max_value=starting_cash + 50*minimum_bet,
                          value=starting_cash + 10*minimum_bet,
                          help="Target amount of cash the gambler aims to reach.")
    p_win = st.slider('Probability of Winning', 0.0, 1.0, 17/36, format="%.2f", help="The gambler's chance of winning a single bet.")

run_sim = st.button('Run Simulation')

# Enhanced Visualization Function
def visualize_current_state(current_state: np.ndarray):
    fig, ax = plt.subplots()
    colors = plt.cm.viridis(current_state / current_state.max())
    bars = ax.bar(np.arange(len(current_state)), current_state, color=colors)
    ax.set_title("Current State of the Gambler's Ruin")
    ax.set_xlabel('Cash Amount')
    ax.set_ylabel('Probability')
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax, label='Probability Density')
    st.pyplot(fig)

if run_sim:
    current_state = run_gamblers_ruin(starting_cash, minimum_bet, goal_cash, p_win)
    visualize_current_state(current_state)
    prob_ruin, prob_success = current_state[0], current_state[-1]
    expected_value = find_expected_value(np.arange(0, len(current_state)), current_state)
    st.metric(label="Expected Value", value=f"{expected_value:.2f}", delta=None)
    st.metric(label="Probability of Ruin", value=f"{prob_ruin:.2%}", delta=None)
    st.metric(label="Probability of Success", value=f"{prob_success:.2%}", delta=None)

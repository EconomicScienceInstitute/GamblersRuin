import streamlit as st
import numpy as np
import plotly.graph_objects as plt
import matplotlib.cm as cm
from src.gamblers_ruin import (create_policy_function, find_nth_state,
                               find_expected_value, run_gamblers_ruin, create_state_map)

# more expansive page config
st.set_page_config(page_title="Gambler's Ruin Simulator", layout="wide")
font = "sans serif"

# Theme switch for light or dark mode
theme = st.sidebar.radio(
    "Choose Font Color", ('Default', 'Red', 'Green', 'Blue'))
# Change the theme of the page bc why not
if theme == 'Blue':
    primaryColor = "#F1F2F6"
    backgroundColor = "#00172B"
    secondaryBackgroundColor = "#0083B8"
    textColor = "#0077FF"
elif theme == 'Green':
    primaryColor = "#F1F2F6"
    backgroundColor = "#00172B"
    secondaryBackgroundColor = "#0083B8"
    textColor = "#009000"
elif theme == 'Red':
    primaryColor = "#F1F2F6"
    backgroundColor = "#00172B"
    secondaryBackgroundColor = "#0083B8"
    textColor = "#FF0031"
else:
    primaryColor = "#F1F2F6"
    backgroundColor = "#00172B"
    secondaryBackgroundColor = "#0083B8"
    textColor = "#000000"


# lil bit of css to make it look nice
st.markdown(
    f"""
    <style>
    .reportview-container {{
        font-family: "sans serif";
        background-color: {backgroundColor};
    }}
    .sidebar .sidebar-content {{
        background-color: {secondaryBackgroundColor};
    }}
    h1, h2, h3, h4, h5, h6, p, li, label, .stButton>button {{
        color: {textColor};
    }}
    .stSlider>div>div>div:nth-child(2), .stButton>button {{
        background-color: {primaryColor};
    }}
    .stButton>button:hover {{
        border: 2px solid {textColor};
        color: {primaryColor};
    }}
    /* Cartoon dogs walking animation */
    @keyframes walk {{
        0% {{ background-position: 0 bottom; }}
        100% {{ background-position: 100% bottom; }}
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# Use columns for a cleaner layout
st.title('Gambler\'s Ruin')
st.markdown('A simple app to simulate the gambler\'s ruin problem')

# Use columns and container for a cleaner layout
with st.sidebar:
    st.header("Simulation Parameters", anchor=None)
    starting_cash = st.slider('Starting Cash', 0, 1000, 500,
                              help="Initial amount of money the gambler starts with.")
    minimum_bet = st.slider('Minimum Bet', 1, 100, 50,
                            help="The smallest amount that can be wagered.")
    goal_cash = st.slider('Goal Cash',
                          min_value=starting_cash,
                          max_value=starting_cash + 50*minimum_bet,
                          value=starting_cash + 10*minimum_bet,
                          help="Target amount of cash the gambler aims to reach.")
    p_win = st.slider('Probability of Winning', 0.0, 1.0, 17/36,
                      format="%.2f", help="The gambler's chance of winning a single bet.")
    periods = st.slider('Number of Periods', 1, 300, 50,
                        help="The amount of periods for the gambler.")

# added a help button
run_sim = st.button('Run Simulation', help="Click to start the simulation.")

# Enhanced Visualization Function


def visualize_current_state(current_state: np.ndarray):
    fig = plt.Figure(
        data=[plt.Bar(x=np.arange(len(current_state)), y=current_state)])
    fig.update_layout(
        title="Current State of the Gambler's Ruin",
        xaxis_title='Cash Amount',
        yaxis_title='Probability',
        coloraxis_colorbar=dict(title='Probability Density',),
        plot_bgcolor=backgroundColor,
        paper_bgcolor=backgroundColor,
        font=dict(color=textColor),
    )
    st.plotly_chart(fig)


if run_sim:
    num_periods = periods
    current_state = run_gamblers_ruin(starting_cash, minimum_bet, goal_cash,
                                      p_win, num_periods)
    visualize_current_state(current_state)
    prob_ruin, prob_success = current_state[0], current_state[1]
    state_map = create_state_map(starting_cash, minimum_bet, goal_cash)
    expected_value = find_expected_value(state_map[0], current_state)
    st.metric(label="Expected Value",
              value=f"{expected_value:.2f}", delta=None)
    st.metric(label="Probability of Ruin",
              value=f"{prob_ruin:.2%}", delta=None)
    st.metric(label="Probability of Success",
              value=f"{prob_success:.2%}", delta=None)

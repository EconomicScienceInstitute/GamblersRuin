import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from gamblers_ruin import (create_policy_function, find_nth_state, 
                           find_expected_value, run_gamblers_ruin, create_state_map)

# I don't think we need this function anymore
def visualize_current_state(current_state: np.ndarray, state_map: np.ndarray):
    fig, ax = plt.subplots()
    colors = cm.viridis(current_state / current_state.max())
    bars = ax.bar(state_map[0], current_state, color=colors,width=50)
    ax.set_title("Current State of the Gambler's Ruin")
    ax.set_xlabel('Cash Amount')
    ax.set_ylabel('Probability')
    return(fig)



def visualize_distribution_over_time(starting_cash, minimum_bet, goal_cash, p_win, max_periods, state_map):
    fig = go.Figure()

    for period in range(1, max_periods + 1):
        state_distribution = run_gamblers_ruin(starting_cash, minimum_bet, goal_cash, p_win, period)
        prob_ruin = state_distribution[0]
        prob_success = state_distribution[-1]

        color_value = 0.5 + prob_success - (prob_ruin*1.5)

        # Map the color_value to a color on the red-green scale
        #color = plt.cm.RdYlGn((color_value + 1) / 2)  # Normalize to [0,1] for the colormap
        #color = f'rgba({color[0]*255}, {color[1]*255}, {color[2]*255}, {color[3]})'

        normalized_color_value = (color_value + 1) / 2
        color = plt.cm.RdYlGn(normalized_color_value)
        rgba_color = f'rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, {color[3]})'


        cum_probs = np.cumsum(state_distribution)
        # Determine the quartile cash values based on the state distribution
        q1_idx, median_idx, q3_idx = np.searchsorted(cum_probs, [0.25, 0.5, 0.75])
        q1_cash, median_cash, q3_cash = state_map[[q1_idx, median_idx, q3_idx]]

        # Use these indices to get the corresponding cash amounts for the quartiles
        lowerfence_cash = state_map[0]
        upperfence_cash = state_map[-1]


        # Add the boxplot for this period
        fig.add_trace(go.Box(
            x=[period] * 5,  # Repeat the period for each y value to align them vertically
            y=[lowerfence_cash, q1_cash, median_cash, q3_cash, upperfence_cash],
            name=f"Period {period}",
            #marker=dict(color=color),  # Set the color of the markers (such as outliers)
            fillcolor=rgba_color,  # Set the fill color of the boxes
            boxpoints=False,
            pointpos=-1.8, 
            line=dict(color='gray', width=1.5),  # Set the line color and width of the quartile markers color='blue'
        ))

    fig.update_layout(
        title='Distribution of Cash Levels Over Time',
        xaxis_title='Period',
        yaxis_title='Cash Amount',
        showlegend=False,
        xaxis = dict(
            tickmode = 'linear',
            tick0 = 0,  # This is the starting tick value. Adjust as needed.
            dtick = 5  # This is the interval between ticks. Adjust to your desired increment.
        ),
        yaxis = dict(
            tickmode = 'linear',
            tick0 = 0,  # This is the starting tick value. Adjust as needed.
            dtick = 100  # This is the interval between ticks. Adjust to your desired increment.
        ),
    )

    return fig

#check if the function works locally
if __name__ == "__main__":
    starting_cash = 500
    minimum_bet = 50
    goal_cash = 1000
    p_win = 0.47
    max_periods = 30
    state_map, start_idx = create_state_map(starting_cash, minimum_bet, goal_cash)


    box = visualize_distribution_over_time(starting_cash, minimum_bet, goal_cash, p_win, max_periods, state_map)
    box.show()

   
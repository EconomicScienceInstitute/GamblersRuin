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
    cash_values = state_map[0]
    colors = cm.viridis(current_state / current_state.max())
    bars = ax.bar(range(len(cash_values)), current_state, color=colors,width=1)
    ax.set_title("Current State of the Gambler's Ruin")
    ax.set_xlabel('Cash Amount')
    ax.set_ylabel('Probability')
    ax.set_xticks(range(len(cash_values)))  # Set x-ticks to match the number of states
    ax.set_xticklabels(cash_values, rotation=90)  # Set x-tick labels as cash values, rotated for readability
    return(fig)


def visualize_current_state_plotly(current_state: np.ndarray, state_map: np.ndarray):
    # Assuming state_map is a tuple with the first element being the array of cash values
    cash_values = state_map[0]
    
    # Create a bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=cash_values,
            y=current_state,
            hoverinfo = 'text',
            hovertemplate='Cash Amount: %{x}<br>Probability: %{y:.2%}<extra></extra>',
            marker=dict(
                color='rgba(255, 0, 0, 0.8)',  # Red color with 80% opacity
            )
        )
    ])
    
    # Customize layout
    fig.update_layout(
        title="Probabilities in Current State of Gambler's Ruin",
        xaxis_title="Cash Amount",
        yaxis_title="Probability",
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=100,  # Set x-axis increment
        ),
        yaxis=dict(
            tickformat=".2%",  # Format the y-axis ticks as percentage
        )
    )
    
    return fig


def create_simulation_df(starting_cash, minimum_bet, goal_cash, p_win, max_periods):
    """Creates a dataframe to store simulation data for multiple periods 
    making it easier to use Plotly.
    """
    # Prepare the state map for the cash levels
    state_map, start_idx = create_state_map(starting_cash, minimum_bet, goal_cash)
    
    # Initialize a list to store the data for each period
    data = []

    for period in range(1, max_periods + 1):
        state_distribution = run_gamblers_ruin(starting_cash, minimum_bet, goal_cash, p_win, period)
        
        # Calculate cumulative probabilities to find quartiles
        cum_probs = np.cumsum(state_distribution)
        q1_idx, median_idx, q3_idx = np.searchsorted(cum_probs, [0.25, 0.5, 0.75])
        q1_cash, median_cash, q3_cash = state_map[[q1_idx, median_idx, q3_idx]]
        
        # Calculate probabilities of ruin and success
        prob_ruin = state_distribution[0]
        prob_success = state_distribution[-1]

        # Calculate the expected value
        expected_value = find_expected_value(state_map, state_distribution)
        
        # Append the data for this period to the list
        data.append({
            'Period': period,
            '25%': q1_cash,
            '50%': median_cash,
            '75%': q3_cash,
            'Prob Success': prob_success,
            'Prob Failure': prob_ruin,
            'Expected Value': expected_value
        })

    # Create a DataFrame from the collected data
    return pd.DataFrame(data)

def visualize_distribution(df, goal_amount):
    fig = go.Figure()

    default_opacity = 1

    cool_colors = {'75%': f'rgba(122,230,167,{default_opacity})', 
                   '50%': f'rgba(128,202,239,{default_opacity})', 
                   '25%': f'rgba(205,146,202,{default_opacity})'}
    
    warm_colors = {'75%': f'rgba(240,232,101,{default_opacity})', 
                   '50%': f'rgba(240,179,101,{default_opacity})', 
                   '25%': f'rgba(240,107,101,{default_opacity})'}
    
    hover_template = ('Round: %{x}<br>' 
                      '% Success: %{customdata[0]:.2%}<br>' +
                      '% Failure: %{customdata[1]:.2%}<br>' +
                      'Expected Value: %{customdata[2]:.2f}<br>' +
                      '%{text} of cases with cash less than %{y}<extra></extra>')
    
    # Add bars for the quartiles
    for col in ['75%', '50%', '25%']:
        fig.add_trace(go.Bar(
            x=df['Period'],
            y=df[col],
            name=col,
            marker_color=warm_colors[col],
            text=[col] * len(df),  # Set the text to display the column name in the hover
            hoverinfo='text',  # Use the text for the hover information
            hovertemplate=hover_template,
            customdata=np.stack((df['Prob Success'], df['Prob Failure'], df['Expected Value']), axis=-1),  # Add custom data for success and failure probabilities
            textposition='none',  # Hide the text labels on the bars
            width=0.8,  # Adjust the width of the bars
            hoverlabel=dict(font=dict(size=14, color='black', family = 'Arial, bold'), bgcolor=warm_colors[col])
        ))

    # Customize the layout
    fig.update_layout(
        title='Percent of Cases with Cash Less Than Values Over Time',
        xaxis=dict(
            title='Period',
            tickmode='linear',
            tick0=0,
            dtick=5,  # Set x-axis increment
            range=[0.5, df['Period'].max() + 0.5]  # Set the range to fully display the first bar
        ),
        yaxis=dict(
            title='Cash Amount',
            tickmode='linear',
            tick0=0,
            dtick=100,  # Set y-axis increment
            range=[0, goal_amount]
        ),
        legend_title='Percentile Limits',
        barmode='overlay'  # This setting will overlay the bars on top of each other
    )

    return fig

#check if the function works locally
if __name__ == "__main__":
    starting_cash = 500
    minimum_bet = 50
    goal_cash = 1000
    p_win = 0.47
    max_periods = 10

    #df = create_simulation_df(starting_cash, minimum_bet, goal_cash, p_win, max_periods)
    #print(df.head(40))
    #fig = visualize_distribution(df, goal_cash)

    current_state = run_gamblers_ruin(starting_cash, minimum_bet, goal_cash, p_win, max_periods)
    state_map = create_state_map(starting_cash, minimum_bet, goal_cash)
    fig = visualize_current_state_plotly(current_state, state_map)

    fig.show()


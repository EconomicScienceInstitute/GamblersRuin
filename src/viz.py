import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as plt
from gamblers_ruin import (create_policy_function, find_nth_state,
                           find_expected_value, run_gamblers_ruin, create_state_map)


def create_data_frame(start_cash: int, min_bet: int, prob_win: float, goal_cash: int, num_rounds: int = 100):
    """
    Simulate the Gambler's Ruin problem and record the state of the gambler's cash at each round.

    Parameters:
        start_cash (int): The initial amount of cash the gambler has.
        min_bet (int): The minimum bet the gambler can place in each round.
        prob_win (float): The probability of winning each bet.
        goal_cash (int): The target amount of cash the gambler aims to reach.
        num_rounds (int): The number of rounds to simulate. Default is 100.

    Returns:
        pd.DataFrame: A DataFrame containing the round number, the gambler's cash at each round, 
                      and the corresponding probability of reaching that cash amount.
    """
    cash = start_cash
    results = []

    # Create state map and initial state
    state_map, start_idx = create_state_map(start_cash, min_bet, goal_cash)
    initial_state = np.zeros(state_map.size)
    initial_state[start_idx] = 1.0

    # Create the transition matrix
    transition_matrix = create_policy_function(state_map.size, prob_win)

    # Simulate the rounds
    for round_number in range(1, num_rounds + 1):
        if cash <= 0 or cash >= goal_cash:
            break

        # Record the current cash and the corresponding probability
        current_state = find_nth_state(
            transition_matrix, initial_state, round_number)
        results.append((round_number, cash, current_state))

        # Update the cash for the next round
        cash = find_expected_value(state_map[0], current_state)
    # Append the final cash and the corresponding probability
    results.append((round_number, cash, current_state))

    # Create DataFrame
    df = pd.DataFrame(results, columns=['Round', 'Cash', 'Probability'])
    return df


def create_animation(start_cash: int, min_bet: int, prob_win: float, goal_cash: int, num_rounds: int = 100):
    """Create plotly animation for Gambler's Ruin simulation.
    Parameters
    ----------------------
    df - dataframe of gambler's ruin simulation
    """

    df = create_data_frame(start_cash, min_bet,
                           prob_win, goal_cash, num_rounds)

    fig = px.bar(df, x='Cash', y='Probability', animation_frame='Round',
                 range_x=[0, goal_cash], range_y=[0, 1],
                 labels={'Cash': 'Cash Amount', 'Probability': 'Probability'})

    fig.update_layout(title='Gambler\'s Ruin Simulation')

    return fig


def simulate_gamblers_ruin_animation(start_cash: int, min_bet: int, goal_cash: int, p: float, periods: int):
    # Initialize arrays to store state probabilities
    state_probabilities_list = []

    # Run the simulation for each period
    for period in range(1, periods + 2):
        # Run the simulation for the current period
        state_probabilities = run_gamblers_ruin(
            start_cash, min_bet, goal_cash, p, period-1)
        state_probabilities_list.append(state_probabilities)

    # Create state map for x-axis labels
    state_map, _ = create_state_map(start_cash, min_bet, goal_cash)

    # Create DataFrame for Plotly Express
    df = pd.DataFrame(state_probabilities_list)

    df = pd.melt(df.reset_index(), id_vars=[
                 'index'], var_name='Cash', value_name='Probability')
    print(df)
    # Melt DataFrame to have 'Cash' and 'Probability' columns and "index" for the frame

    df['Cash'] = state_map[df['Cash'].astype(int)]
    print(df)
    # Create animated bar chart
    fig = px.bar(df, x='Cash', y='Probability', animation_frame=df['index'] + 1,
                 title='Gambler\'s Ruin Simulation')

    # Set x-axis and y-axis labels
    fig.update_layout(xaxis_title='Cash Amount', yaxis_title='Probability')

    return fig


if __name__ == "__main__":
    fig = simulate_gamblers_ruin_animation(500, 50, 1000, 0.47, 50)
    fig.show()

import pandas as pd
import plotly.express as px
from gamblers_ruin import (create_policy_function, find_nth_state,
                           find_expected_value, run_gamblers_ruin, create_state_map)


def run_gamblers_ruin_animation(start_cash: int, min_bet: int, goal_cash: int, p: float, periods: int):
    """
    Create plotly animation for Gambler's Ruin simulation.

     Parameters:
        start_cash (int): The initial amount of cash the gambler has.
        min_bet (int): The minimum bet the gambler can place in each round.
        prob_win (float): The probability of winning each bet.
        goal_cash (int): The target amount of cash the gambler aims to reach.
        num_rounds (int): The number of rounds to simulate. Default is 100.

    Returns:
        figure
    """
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

    df = df.rename(columns={'index': 'Round #'})

    df['Cash'] = state_map[df['Cash'].astype(int)]
    # Create animated bar chart
    fig = px.bar(df, x='Cash', y='Probability', animation_frame=df['Round #'],
                 title='Gambler\'s Ruin Animation', color='Probability',
                 color_continuous_scale='viridis')

    fig.update_layout(xaxis_title='Cash Amount', yaxis_title='Probability')

    return fig


# for testing
if __name__ == "__main__":
    fig = run_gamblers_ruin_animation(500, 50, 1000, 0.47, 50)
    fig.show()

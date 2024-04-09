import plotly.express as px
from gamblers_ruin import (create_state_map, run_gamblers_ruin)
import pandas as pd

def create_data_frame(start_cash: int, min_bet: int,
                      prob_win: float, goal_cash: int,
                      num_rounds: int=100):
    """Create a DataFrame of the gambler's ruin simulation in a shape that is
    suitable for Plotly animation.

    Parameters
    ----------
    start_cash
        Amount of cash that our gambler starts with
    min_bet
        Minimum bet that the gambler can make
    prob_win
        Probability of winning a bet
    goal_cash
        Amount of cash that the gambler wants to reach
    num_rounds
        Number of rounds to simulate
    """

    # lazy way to create one state probability vector for each round
    for i in range(num_rounds):
        if i == 0:
            current_state = run_gamblers_ruin(start_cash, min_bet, goal_cash, prob_win, i)
            df = pd.DataFrame(current_state, columns=[i])
        else:
            current_state = run_gamblers_ruin(start_cash, min_bet, goal_cash, prob_win, i)
            df[i] = current_state
    df = df.transpose() # need to transpose becasue we did it vertical the lazy way
    state_map, _ = create_state_map(start_cash, min_bet, goal_cash)
    df.columns = state_map # state map is the cash values
    df["Round"] = df.index # index should correspond to the round can do +1
    return df.melt(id_vars="Round",
                  var_name = "Cash",
                  value_name = "Probability")

def create_animation(start_cash: int, min_bet: int,
                      prob_win: float, goal_cash: int,
                      num_rounds: int=100):
    """Create a Plotly visualization of the gambler's ruin simulation.

    Parameters
    ----------
    df
        DataFrame of the gambler's ruin simulation
    """
    df = create_data_frame(start_cash, min_bet,
                           prob_win, goal_cash,
                           num_rounds)
    fig = px.bar(df, x="Cash", y="Probability",
                  title="Gambler's Ruin Simulation",
                  animation_frame="Round",
                  range_y=[0, 1])
    return fig

if __name__ == "__main__":
    fig = create_animation(100, 10, 0.5, 200, 100)
    fig.show()
    # should be able to futher visualize the data with streamlit in our main
    # st.plotly_chart(fig, use_container_width=True)
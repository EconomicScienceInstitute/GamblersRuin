# Gamblers Ruin
Implementation of the gamblers ruin

In our case a man walks into a casino with $500 cash.

He decides to play roulette at a table where the minimum bet is $50.

His policy function is defined as follows:

    If I have more than my initial amount, I will place the minimum bet.
    If I have less than my minimum amount I will bet enough so that I will have $550 dollars if I win.

If his cash on hand reaches $0 he goes bankrupt and is done.

If his cash on hand reaches $1000 he leaves the casino.

We will answer several questions about this environment:

1. What is the probability of him going bankrupt after `n` rounds?

2. What is the probability of him leaving the casino with $1000 after `n` rounds?

3. What is the expected value of his cash on hand after `n` rounds?

## Prerequisite steps

`pip install -r requirements.txt`

## How to Run
To run in normal mode with no debugging run the following code in your console
`streamlit run src/main.py`

To run with the debugger use the `Python: Strealmit Debug` launch configuration while the file `src/main.py` is open.

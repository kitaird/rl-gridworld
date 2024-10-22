import matplotlib.pyplot as plt
import pandas as pd


def plot_returns(algo_name, returns):
    n_steps = len(returns) // 10
    df = pd.DataFrame(returns)
    line = df.rolling(n_steps).mean()

    sigma_factor = 2
    ci = sigma_factor * df.rolling(n_steps).std()
    upper_line = (line + ci)[0]
    lower_bound = (line - ci)[0]

    plt.plot(line, linewidth=2, label=f"Avg. over {n_steps} episodes")
    plt.fill_between(ci.index, lower_bound, upper_line, color='red', alpha=.3, label=f'{sigma_factor}-Sigma CI')

    plt.xlabel("Episodes")
    plt.ylabel("Return")
    plt.title(f"Episode Returns: {algo_name}")
    plt.legend()
    plt.show()


def plot_value_function_sums(algo_name, value_functions_sum):
    plt.plot(value_functions_sum, label="Value Function Sum")
    plt.xlabel("Iterations")
    plt.ylabel("Value Function Sum")
    plt.legend()
    plt.title(f"Value Functions: {algo_name}")
    plt.show()

# %%
# Data processing
import numpy as np
import pandas as pd
import datetime as dt

# Graphing
from matplotlib import pyplot as plt

from tqdm import tqdm
# %%
# Initial investment and currency per stock in dollars
investment = {
    "^HSI": [500_000, 'HKD'],  # Hang Seng Index
    "^AEX": [500_000, 'EUR'],  # AEX Index
    "^GSPC": [500_000, 'USD'],  # S&P 500 Index
    "^N225": [500_000, 'JPY']  # Nikkei 225
}

relative_weights = np.array([0.5, 0.5, 0.5, 0.5, -1.0])
absolute_weights = np.array([500_000, 500_000, 500_000, 500_000, -1_000_000])

df_losses = pd.read_csv('A1/Data/LinearLosses.csv',
                        parse_dates=True, index_col=0, )
# %%


def empirical_var(x, alpha):
    return np.quantile(np.sort(x), alpha, interpolation='linear')


def empirical_es(x, alpha):
    var = empirical_var(x, alpha)
    return x[x >= var].mean()

# %%


def VaR_HS(absolute_weights: np.ndarray, df_returns: pd.DataFrame, windows: list) -> pd.DataFrame:
    """Creates Value at Risk figures for Historical Simulation method
    Args:
        absolute_weights (np.ndarray): array with weights 
        df_returns (pd.DataFrame): dataframe with returns
    """
    df = pd.DataFrame(index=df_returns.index)
    df['Returns'] = np.sum(df_returns * absolute_weights, axis=1)
    alphas = [0.975, 0.990]

    for alpha in alphas:
        for window in windows:
            df[f'VaR_HS_{window}_{alpha}_N'] = np.NaN
            df[f'ES_HS_{window}_{alpha}_N'] = np.NaN
            for t in tqdm(range(window, len(df_returns))):
                df[f'VaR_HS_{window}_{alpha}_N'].iloc[t] = empirical_var(
                    df['Returns'].iloc[t-window:t], alpha)
                df[f'ES_HS_{window}_{alpha}_N'].iloc[t] = empirical_es(
                    df['Returns'].iloc[t-window:t], alpha)

    return df


# %%
windows = [300, 1000, 2000]
df_var = VaR_HS(absolute_weights, df_losses, windows)

# %%
df_var.to_csv('A1/Data/VaR/Historical-Simulation.csv')

# %%

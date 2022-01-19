# %%
# Data processing
import numpy as np
import pandas as pd
import datetime as dt

# Graphing
from matplotlib import pyplot as plt

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

# Get prices from start date till end date
start = "2010-01-01"
end = "2021-04-01"

# %%
df_returns = pd.read_csv('A1/Data/Returns.csv', index_col=0)
# %%
# Linearize losses
losses = pd.DataFrame(index=df_returns.index)

losses['AEX'] = df_returns['AEX']
losses[['GSPC', 'HSI', 'N225']] = 1 - \
    np.exp(df_returns[['GSPC', 'HSI', 'N225']].values -
           df_returns[['EURUSD', 'EURHKD', 'EURJPY']].values)
losses = losses[1:]
losses['Time-to-Maturity'] = pd.Series(np.arange(20,
                                                 0, -1/(250))[:len(df_returns)-1]).values


losses['Interest Payment'] = df_returns['Libor']/(250*100)
losses['Bond Price'] = 1 - \
    np.exp(-losses['Time-to-Maturity'] * (df_returns['Change Libor']/100))

losses['Bond'] = losses['Interest Payment'] - losses['Bond Price']
losses = losses[['AEX', 'GSPC', 'HSI', 'N225', 'Bond']]
losses = losses.dropna()
losses.to_csv('A1/Data/LinearLosses.csv')

# %%
total_losses = (losses * absolute_weights).sum(axis=1)
total_losses.to_csv('A1/Data/HistoricalLosses.csv')

# %%

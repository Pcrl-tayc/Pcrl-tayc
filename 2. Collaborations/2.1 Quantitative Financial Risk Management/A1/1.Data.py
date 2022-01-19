# %%
# Data processing
import numpy as np
import pandas as pd
import datetime as dt

# Download data
import yfinance as yf

# Graphing
from matplotlib import pyplot as plt

from functools import reduce

# %%
# Initial investment and currency per stock in dollars

investment = {
    "^HSI": [500_000, 'HKD'],  # Hang Seng Index
    "^AEX": [500_000, 'EUR'],  # AEX Index
    "^GSPC": [500_000, 'USD'],  # S&P 500 Index
    "^N225": [500_000, 'JPY']  # Nikkei 225
}

relative_weights = [0.5, 0.5, 0.5, 0.5, -1.0]
absolute_weights = [500_000, 500_000, 500_000, 500_000, -1_000_000]

# Get prices from start date till end date
start = "2010-01-01"
end = "2021-04-01"

# %%
# 3-Month London Interbank Offered Rate (LIBOR), based on Euro
# source: https://fred.stlouisfed.org/series/EUR3MTD156N

libor_eur = pd.read_csv('A1/Data/EUR3MTD156N.csv', names=[
                        'Date', '3M Libor'], header=0, parse_dates=True, index_col='Date', na_values='.', dtype={'3M Libor': np.float64})
libor_eur = libor_eur[start:end]
# %%
# Download data and store in dataframe
instruments_data = yf.download(
    ' '.join(list(investment.keys())), start=start, end=end, groupby='ticker')
instruments = instruments_data['Adj Close']
# %%
# Create exchange rate ticker list based on initial investment data
currencies = list(set(['EUR' + i[1] + '=X' for i in investment.values()]))

# Remove EUR-EUR exchange rate
for currency in currencies:
    if currency[0:2] == currency[3:5]:
        currencies.remove(currency)

currencies = " ".join(currencies)


# Download exchange rate data
exchangerate_data = yf.download(
    currencies, start=start, end=end, groupby='ticker')
exchangerate = exchangerate_data['Adj Close']
# %%
# Join all instruments
dfs = [instruments, exchangerate, libor_eur]
df_final = reduce(lambda left, right: pd.merge(
    left, right, left_index=True, right_index=True), dfs)
df_final = df_final.rename(columns={'^AEX': 'AEX', '^GSPC': 'GSPC', '^HSI': 'HSI',
                                    '^N225': 'N225', 'EURHKD=X': 'EURHKD', 'EURJPY=X': 'EURJPY', 'EURUSD=X': 'EURUSD'})
# %%
# Export prices to csv
df_final.to_csv('A1/Data/MarketPrices.csv')

# %%
# Calculate currency adjusted returns of assets
stock_returns = np.log(df_final.iloc[:, 0:7]).diff()
stock_returns['Libor'] = (df_final['3M Libor'])
stock_returns['Change Libor'] = stock_returns['Libor'].diff()


# stock_returns[['GSPC', 'HSI', 'N225']] = stock_returns[['GSPC', 'HSI', 'N225']
                                                    #    ].values - np.log(df_final[['EURUSD', 'EURHKD', 'EURJPY']]).diff().values
# yields = pd.DataFrame(index=df_final.index)
# yields['3M Libor'] = (df_final['3M Libor'] / 100)
# yields['Price'] = 1/((1+yields['3M Libor'])**(3/12))
# yields['Yields']= np.log(yields['Price']).diff()
# %%
# df_returns = pd.merge(
    # stock_returns, yields['Yields'], left_index=True, right_index=True).dropna()
stock_returns.to_csv('A1/Data/Returns.csv')

# %%
plt.style.use("seaborn-muted")

for stock in stock_returns.columns[:4]:
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))

    df_final[stock].plot(ax=ax[0], grid=True,
                         title=f'{stock} prices')  # row=0, col=0
    ax[0].set(xlabel=None)
    stock_returns[stock].plot(
        ax=ax[1], grid=True, title=f'{stock} log-returns')  # row=0, col=1
    ax[1].set(xlabel=None)
    plt.subplots_adjust()
    plt.tight_layout()
    plt.savefig(f'A1/Figures/{stock}_descriptive.eps', format='eps')
# %%

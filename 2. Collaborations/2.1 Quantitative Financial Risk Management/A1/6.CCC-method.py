# %%
# Data processing
from scipy.stats import norm
import warnings
import numpy as np
import pandas as pd
import datetime as dt

# Graphing
from matplotlib import pyplot as plt

# Arch
from arch import arch_model

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import collections
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

df_losses = pd.read_csv('A1/Data/LinearLosses.csv',
                        parse_dates=True, index_col=0, )

split_dates = ['2010-01-01', '2010-12-31', '2011-12-31', '2012-12-31', '2013-12-31',
               '2014-12-31', '2015-12-31', '2016-12-31', '2017-12-31', '2018-12-31', '2019-12-31', '2020-12-31', "2021-04-01"]

# %%
# pf_std = np.sqrt(np.dot(absolute_weights.T, np.dot(H, absolute_weights)))
# %%
split_dates = ['2010-01-01', '2010-12-31', '2011-12-31', '2012-12-31', '2013-12-31',
               '2014-12-31', '2015-12-31', '2016-12-31', '2017-12-31', '2018-12-31', '2019-12-31', '2020-12-31', "2021-04-01"]

# This kind of dict lets you initiate a key and append to it simultaneously
rolling_var = collections.defaultdict(list)

rolling_es = collections.defaultdict(list)

# Store forecasted conditional volatilities on a rolling window basis
rolling_forecasted_volatilities = pd.DataFrame()

# Store rolling correlation matrix
rolling_corr_matrix = collections.defaultdict(list)

# Year 1 and 2 are used to estimate the VaR for year 3
# Year 2 and 3 are used to estimate the VaR for year 4
# Etc.

df_returns = df_losses * absolute_weights
weights = absolute_weights
# %%

# This kind of dict lets you initiate a key and append to it simultaneously
rolling_var = collections.defaultdict(list)

rolling_es = collections.defaultdict(list)

# Store forecasted conditional volatilities on a rolling window basis
rolling_forecasted_volatilities = pd.DataFrame()

# Store rolling correlation matrix
rolling_corr_matrix = collections.defaultdict(list)

# Year 1 and 2 are used to estimate the VaR for year 3
# Year 2 and 3 are used to estimate the VaR for year 4
# Etc.

df_returns = df_losses * absolute_weights

for i in range(len(split_dates) - 3):

    # Forecast volatilities with a rolling window using a GARCH(1,1)
    forecasted_volatility = pd.DataFrame(
        index=df_returns[split_dates[i+2]:split_dates[i+3]].index)

    for instrument in df_losses.columns:

        garch11 = arch_model(df_losses[instrument],
                             p=1, q=1, dist='Normal', vol='Garch', mean='Zero')
        res = garch11.fit(
            update_freq=10, last_obs=split_dates[i+2], disp='off')
        forecasts = res.forecast(horizon=1, start=split_dates[i+2])
        forecasted_variance = forecasts.variance[split_dates[i+2]
            :split_dates[i+3]]
        forecasted_volatility[instrument] = np.sqrt(forecasted_variance)

    # Append the forecasted year to the rolling_forecasted_volatilities dataframe
    rolling_forecasted_volatilities = rolling_forecasted_volatilities.append(
        forecasted_volatility)

    # Append rolling correlation matrix to list
    rolling_corr = df_returns[split_dates[i]:split_dates[i+2]].corr()
    rolling_corr_matrix[split_dates[i+3]].append(rolling_corr)


# %%


# %%
# Set up lists containing the confidence levels
confidence_levels = [0.975, 0.99]

# This kind of dict lets you initiate a key and append to it simultaneously
rolling_var = collections.defaultdict(list)

rolling_es = collections.defaultdict(list)

# Store forecasted conditional volatilities on a rolling window basis
rolling_forecasted_volatilities = pd.DataFrame()

# Store rolling correlation matrix
rolling_corr_matrix = collections.defaultdict(list)

# Year 1 and 2 are used to estimate the VaR for year 3
# Year 2 and 3 are used to estimate the VaR for year 4
# Etc.

df_returns = df_losses * absolute_weights

for i in range(len(split_dates) - 3):

    # Forecast volatilities with a rolling window using a GARCH(1,1)
    forecasted_volatility = pd.DataFrame(
        index=df_returns[split_dates[i+2]:split_dates[i+3]].index)

    for instrument in df_losses.columns:

        garch11 = arch_model(df_losses[instrument],
                             p=1, q=1, dist='Normal', vol='Garch', mean='Zero')
        res = garch11.fit(
            update_freq=10, last_obs=split_dates[i+2], disp='off')
        forecasts = res.forecast(horizon=1, start=split_dates[i+2])
        forecasted_variance = forecasts.variance[split_dates[i+2]
            :split_dates[i+3]]
        forecasted_volatility[instrument] = np.sqrt(forecasted_variance)

    # Append the forecasted year to the rolling_forecasted_volatilities dataframe
    rolling_forecasted_volatilities = rolling_forecasted_volatilities.append(
        forecasted_volatility)

    # Append rolling correlation matrix to list
    rolling_corr = df_returns[split_dates[i]:split_dates[i+2]].corr()
    rolling_corr_matrix[split_dates[i+3]].append(rolling_corr)


# Use the conditional volatilities and correlations matrices to calculate the VaR and ES
# The approach is similar to the var-covar method in that the covariance matrix is used
# to calculate VaR and ES.


var_975_1d_ccc = []
var_990_1d_ccc = []
es_975_1d_ccc = []
es_990_1d_ccc = []

for index, row in rolling_forecasted_volatilities.iterrows():

    # For each conditional volatility in the dataframe calculate the VaR and ES
    vola = np.array(row)

    # Refresh the correlation matrix each year
    for date in list(rolling_corr_matrix.keys())[0:10]:
        if index < dt.date.fromisoformat(date):
            cons_corr = rolling_corr_matrix.get(date)
            break

    cov = vola * vola * cons_corr[0]
    # print(cov)
    ccc_std_euro = np.sqrt(absolute_weights.T.dot(cov).dot(absolute_weights))

    var_975_1d_ccc_value = VaR_SNorm(0, ccc_std_euro, confidence_levels[0])
    var_990_1d_ccc_value = VaR_SNorm(0, ccc_std_euro, confidence_levels[1])

    es_975_1d_ccc_value = ES_SNorm(0, ccc_std_euro, confidence_levels[0])
    es_990_1d_ccc_value = ES_SNorm(0, ccc_std_euro, confidence_levels[1])

    var_975_1d_ccc.append(var_975_1d_ccc_value)
    var_990_1d_ccc.append(var_990_1d_ccc_value)

    es_975_1d_ccc.append(es_975_1d_ccc_value)
    es_990_1d_ccc.append(es_990_1d_ccc_value)

# Store estimates in dataframe
ccc = pd.DataFrame(data=[var_975_1d_ccc, var_990_1d_ccc,
                         es_975_1d_ccc, es_990_1d_ccc]).transpose()
ccc.columns = ['var_975_1d_ccc', 'var_990_1d_ccc',
               'es_975_1d_ccc', 'es_990_1d_ccc']
ccc = ccc.set_index(rolling_forecasted_volatilities.index)
# ccc['losses'] = np.sum(df_losses.values * absolute_weights, axis=1)[-1819:]

# %%
# Store estimates in dataframe

# %%
rolling_var['var975ccc'].append(ccc['var_975_1d_ccc'])
rolling_var['var990ccc'].append(ccc['var_990_1d_ccc'])

rolling_es['es975ccc'].append(ccc['es_975_1d_ccc'])
rolling_es['es990ccc'].append(ccc['es_990_1d_ccc'])
# %%


# %%
# Functions to calculate VaR and ES
def VaR_SNorm(mean, std, alpha):
    return mean + std * norm.ppf(alpha)


def ES_SNorm(mean, std, alpha):
    return mean + std * norm.pdf(norm.ppf(alpha)) / (1 - alpha)


def CC_VaR(absolute_weights, df_losses, split_dates):
    confidence_levels = [0.975, 0.99]
    # This kind of dict lets you initiate a key and append to it simultaneously
    rolling_var = collections.defaultdict(list)
    rolling_es = collections.defaultdict(list)
    # Store forecasted conditional volatilities on a rolling window basis
    rolling_forecasted_volatilities = pd.DataFrame()
    # Store rolling correlation matrix
    rolling_corr_matrix = collections.defaultdict(list)
    # Year 1 and 2 are used to estimate the VaR for year 3,# Year 2 and 3 are used to estimate the VaR for year 4,# Etc.
    df_returns = df_losses * absolute_weights
    for i in range(len(split_dates) - 3):
        # Forecast volatilities with a rolling window using a GARCH(1,1)
        forecasted_volatility = pd.DataFrame(
            index=df_returns[split_dates[i+2]:split_dates[i+3]].index)
        for instrument in df_losses.columns:
            garch11 = arch_model(df_losses[instrument],
                                 p=1, q=1, dist='Normal', vol='Garch', mean='Zero')
            res = garch11.fit(
                update_freq=10, last_obs=split_dates[i+2], disp='off')
            forecasts = res.forecast(horizon=1, start=split_dates[i+2])
            forecasted_variance = forecasts.variance[split_dates[i+2]
                :split_dates[i+3]]
            forecasted_volatility[instrument] = np.sqrt(forecasted_variance)

        # Append the forecasted year to the rolling_forecasted_volatilities dataframe
        rolling_forecasted_volatilities = rolling_forecasted_volatilities.append(
            forecasted_volatility)
        # Append rolling correlation matrix to list
        rolling_corr = df_returns[split_dates[i]:split_dates[i+2]].corr()
        rolling_corr_matrix[split_dates[i+3]].append(rolling_corr)

    # Use the conditional volatilities and correlations matrices to calculate the VaR and ES,# The approach is similar to the var-covar method in that the covariance matrix is used,# to calculate VaR and ES.
    var_975_1d_ccc = []
    var_990_1d_ccc = []
    es_975_1d_ccc = []
    es_990_1d_ccc = []
    for index, row in rolling_forecasted_volatilities.iterrows():
        # For each conditional volatility in the dataframe calculate the VaR and ES
        vola = np.array(row)
        # Refresh the correlation matrix each year
        for date in list(rolling_corr_matrix.keys())[0:10]:
            if index < dt.date.fromisoformat(date):
                cons_corr = rolling_corr_matrix.get(date)
                break
        cov = vola * vola * cons_corr[0]
        # print(cov)
        ccc_std_euro = np.sqrt(
            absolute_weights.T.dot(cov).dot(absolute_weights))
        var_975_1d_ccc_value = VaR_SNorm(0, ccc_std_euro, confidence_levels[0])
        var_990_1d_ccc_value = VaR_SNorm(0, ccc_std_euro, confidence_levels[1])
        es_975_1d_ccc_value = ES_SNorm(0, ccc_std_euro, confidence_levels[0])
        es_990_1d_ccc_value = ES_SNorm(0, ccc_std_euro, confidence_levels[1])
        var_975_1d_ccc.append(var_975_1d_ccc_value)
        var_990_1d_ccc.append(var_990_1d_ccc_value)
        es_975_1d_ccc.append(es_975_1d_ccc_value)
        es_990_1d_ccc.append(es_990_1d_ccc_value)

    # Store estimates in dataframe
    ccc = pd.DataFrame(data=[var_975_1d_ccc, var_990_1d_ccc,es_975_1d_ccc, es_990_1d_ccc]).transpose()
    ccc.columns = ['var_975_1d_ccc', 'var_990_1d_ccc',
                   'es_975_1d_ccc', 'es_990_1d_ccc']
    
    ccc = ccc.set_index(rolling_forecasted_volatilities.index)

    return ccc
# %%
test = CC_VaR(absolute_weights, df_losses, split_dates)
# %%

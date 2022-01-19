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

# %%
split_dates_ccc = ['2010-01-01', '2010-12-31', '2011-12-31', '2012-12-31', '2013-12-31',
                   '2014-12-31', '2015-12-31', '2016-12-31', '2017-12-31', '2018-12-31', '2019-12-31', '2020-12-31']


def _CalibrateGarch(df_losses):
    p = df_losses.corr()
    factor_models = [arch_model(df_losses[column], mean='zero',
                                p=1, q=1, rescale=True) for column in df_losses.columns]
    results = [model.fit(update_freq=0, disp='off') for model in factor_models]
    scales = [result.scale for result in results]
    forecasts = [result.forecast() for result in results]

    means = np.array([forecast.mean.values[-1, 0] for forecast in forecasts])
    stds = np.array([np.sqrt(forecast.variance.values[-1, 0])
                     for forecast in forecasts])

    means /= scales
    stds /= scales
    diag = np.diag(stds)
    cov = diag.dot(p).dot(diag)
    return means, cov


# %%
# Store forecasted conditional volatilities on a rolling window basis
rolling_forecasted_volatilities = pd.DataFrame()

for i in range(1):
    window = df_losses[split_dates_ccc[i]:split_dates_ccc[i+2]]
    means, cov = _CalibrateGarch(window)


# %%
# calibrate starting parameters


parameters = pd.DataFrame(
    index=['omega', 'alpha', 'beta'], columns=df_losses.columns)
training_window = 600
# for asset in df_losses.columns[0]:
model = arch_model(df_losses['AEX'][:training_window], vol="Garch",
                   p=1, q=1, rescale=True, mean='Zero')
res = model.fit(update_freq=10, disp="off")
parameters['AEX'] = res.params.values.T
forecasts = res.forecast(horizon=100)
means = ([forecasts.mean.values[-1, 0]])
stds = np.sqrt(forecasts.variance.values[-1, 0])
# print(forecasts)
sigma2_0 = np.array(df_losses.iloc[0:50].var())

# factor_models = [arch_model(df_losses[column][:300], mean='Zero', p=1, q=1, rescale=True) for column in df_losses.columns]
# for model in factor_models
# results = [model.fit(update_freq=0, disp='off') for model in factor_models]


# %%
# garch for each asset
training_window = 1000
df_aex = df_losses['AEX']
model = arch_model(df_aex[:training_window], mean='Zero',
                   p=1, q=1,  dist='Normal', vol='Garch')
res = model.fit(update_freq=10, disp="off")
parameters = np.array(res.params.values)


# %%
def _CalibrateGarch(df_losses):
    parameters = {}
    for asset in df_losses.columns:
        model = arch_model(df_losses[asset], mean='Zero',
                           p=1, q=1,  dist='Normal', vol='Garch')
        res = model.fit(update_freq=10, disp="off", show_warning=False)
        parameters[asset] = np.array(res.params.values)
        # print(np.sum(parameters[asset][1:]))
    return parameters


def Garch(df_losses, calibration_window=1000):
    variances = np.zeros(shape=(len(df_losses), len(df_losses.columns)))
    volatility = np.zeros(shape=(len(df_losses), len(df_losses.columns)))
    volatility[0] = 0.01  # 1% starting volatility
    for time in range(1, len(df_losses)):

        if time <= calibration_window:
            parameters = _CalibrateGarch(df_losses[0:calibration_window])

        if (time + calibration_window) % 250 == 0:

            print(
                f'Time to calibrate model at point {time} using data from {0} to {time + calibration_window}')
            parameters = _CalibrateGarch(df_losses[:time + calibration_window])

        omega = [elem[0] for elem in parameters.values()]
        alpha = [elem[1] for elem in parameters.values()]
        beta = [elem[2] for elem in parameters.values()]
        variances[time, :] = np.array(omega +
                                      alpha * (df_losses.iloc[time-1]**2) +
                                      beta * (volatility[time-1]**2))
        volatility[time, :] = np.sqrt(variances[time, :])

    return volatility


# %%
vol = Garch(df_losses, calibration_window=1000)

VaR = np.sum(vol * absolute_weights, axis=1)*1.96
# %%
for i in range(1, len(result)):
    result.loc[i, 'variance'] = result.loc[i-1, 'omega'] + \
        result.loc[i-1, 'alpha'] * result.loc[i-1, 'return']**2 + \
        result.loc[i-1, 'beta'] * result.loc[i-1, 'vola']**2
    result.loc[i, 'vola'] = np.sqrt(result.loc[i, 'variance'])

# %%
    # result.loc[i, 'variance'] = result.loc[i-1, 'omega'] + \
    #     result.loc[i-1, 'alpha'] * result.loc[i-1, 'return']**2 + \
    #     result.loc[i-1, 'beta'] * result.loc[i-1, 'vola']**2

# %%
coeffs = []
cond_vol = []
std_resids = []
models = []

df_test = df_losses
# Estimate the univariate GARCH models:
for asset in df_test.columns:
    model = arch_model(df_test[asset], mean='Zero',
                       vol='GARCH', p=1, o=0,
                       q=1).fit(update_freq=0, disp='off')
    coeffs.append(model.params)
    cond_vol.append(model.conditional_volatility)
    std_resids.append(model.resid / model.conditional_volatility)
    models.append(model)

# Store the results in DataFrames:
coeffs_df = pd.DataFrame(coeffs, index=df_losses.columns)
cond_vol_df = pd.DataFrame(cond_vol).transpose() \
    .set_axis(df_losses.columns,
              axis='columns',
              inplace=False)
std_resids_df = pd.DataFrame(std_resids).transpose() \
    .set_axis(df_losses.columns,
              axis='columns',
              inplace=False)

# %%
# Calculate the constant conditional correlation matrix (R):
R = std_resids_df.transpose() \
    .dot(std_resids_df) \
    .div(len(std_resids_df))

# %%
# Calculate the one-step-ahead forecast of the conditional covariance matrix:
diag = []
N = len(df_losses.columns)
D = np.zeros((N, N))
for model in models:
    diag.append(model.forecast(horizon=1).variance.values[-1][0])
diag = np.sqrt(np.array(diag))
np.fill_diagonal(D, diag)
H = np.matmul(np.matmul(D, R.values), D)

# %%


def normal_var(mean, std, alpha):
    return mean + std * norm.ppf(alpha)


# %%
pf_std = np.sqrt(np.dot(absolute_weights.T, np.dot(H, absolute_weights)))
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

# Temporary list for series of FHS VaRs and ESs
var_975_fhs_rolling = []
var_990_fhs_rolling = []

es_975_fhs_rolling = []
es_990_fhs_rolling = []

# Year 1 and 2 are used to estimate the VaR for year 3
# Year 2 and 3 are used to estimate the VaR for year 4
# Etc.

df_returns = df_losses * absolute_weights
weights = absolute_weights
# %%
for i in range(len(split_dates) - 3):

    # Forecast volatilities with a rolling window using a GARCH(1,1)
    forecasted_volatility = pd.DataFrame(
        index=df_returns[split_dates[i+2]:split_dates[i+3]].index)

    for instrument in df_returns.columns:

        garch11 = arch_model(df_returns[instrument],
                             p=1, q=1, dist='Normal', vol='Garch')
        res = garch11.fit(
            update_freq=10, last_obs=split_dates[i+2], disp='off')
        forecasts = res.forecast(horizon=1, start=split_dates[i+2])
        forecasted_variance = forecasts.variance[split_dates[i+2]:split_dates[i+3]]
        forecasted_volatility[instrument] = np.sqrt(forecasted_variance)

    # Append the forecasted year to the rolling_forecasted_volatilities dataframe
    rolling_forecasted_volatilities = rolling_forecasted_volatilities.append(
        forecasted_volatility)

    # Append rolling correlation matrix to list
    rolling_corr = df_returns[split_dates[i]:split_dates[i+2]].corr()
    rolling_corr_matrix[split_dates[i+3]].append(rolling_corr)


# %%
# Functions to calculate VaR and ES
def VaR_SNorm(mean, std, alpha):
    return mean + std * norm.ppf(alpha)


def ES_SNorm(mean, std, alpha):
    return mean + std * norm.pdf(norm.ppf(alpha)) / (1 - alpha)


# Set up lists containing the confidence levels
confidence_levels = [0.975, 0.99]
# %%
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
    ccc_std_euro = np.sqrt(relative_weights.T.dot(cov).dot(relative_weights))

    var_975_1d_ccc_value = VaR_SNorm(0, ccc_std_euro, confidence_levels[0])
    var_990_1d_ccc_value = VaR_SNorm(0, ccc_std_euro, confidence_levels[1])

    es_975_1d_ccc_value = ES_SNorm(0, ccc_std_euro, confidence_levels[0])
    es_990_1d_ccc_value = ES_SNorm(0, ccc_std_euro, confidence_levels[1])

    var_975_1d_ccc.append(var_975_1d_ccc_value)
    var_990_1d_ccc.append(var_990_1d_ccc_value)

    es_975_1d_ccc.append(es_975_1d_ccc_value)
    es_990_1d_ccc.append(es_990_1d_ccc_value)

# %%
# Store estimates in dataframe
ccc = pd.DataFrame(data=[var_975_1d_ccc, var_990_1d_ccc,
                         es_975_1d_ccc, es_990_1d_ccc]).transpose()
ccc.columns = ['var_975_1d_ccc', 'var_990_1d_ccc',
               'es_975_1d_ccc', 'es_990_1d_ccc']
ccc = ccc.set_index(rolling_forecasted_volatilities.index)
ccc['losses'] = np.sum(df_losses.values * absolute_weights, axis=1)[-1819:]
# %%
rolling_var['var975ccc'].append(ccc['var_975_1d_ccc'])
rolling_var['var990ccc'].append(ccc['var_990_1d_ccc'])

rolling_es['es975ccc'].append(ccc['es_975_1d_ccc'])
rolling_es['es990ccc'].append(ccc['es_990_1d_ccc'])
# %%

# %%
# Data processing
import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import t as studentst
from scipy import stats

# Graphing
from matplotlib import pyplot as plt

# Statistics
from scipy.stats import norm

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

# Get prices from start date till end date
start = "2010-01-01"
end = "2021-04-01"

# %%
df_losses = pd.read_csv('A1/Data/LinearLosses.csv',
                        parse_dates=True, index_col=0, )
# %%
# Functions to calculate VaR and ES


def normal_var(mean, std, alpha):
    return mean + std * norm.ppf(alpha)


def normal_es(mean, std, alpha):
    return mean + std * norm.pdf(norm.ppf(alpha)) / (1 - alpha)


def t_var(v, mean, std, alpha):
    t_std = std / np.sqrt(v / (v-2))
    return mean + std * studentst.ppf(alpha, v)


def t_es(v, mean, std, alpha):
    t_std = std / np.sqrt(v / (v-2))
    return mean + std * studentst.pdf(studentst.ppf(alpha, v), v) / (1 - alpha) * \
        (v + studentst.ppf(alpha, v)**2) / (v-1)


def var_covar(returns, weights):
    mean = returns.mean()
    pf_mean = np.sum(mean*weights)

    cov_matrix = returns.cov()
    pf_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    return pf_mean, pf_std


# %%
def VaR_VC(absolute_weights: np.ndarray, df_returns: pd.DataFrame, windows: list, drop: bool) -> pd.DataFrame:
    """Creates Value at Risk figures for Variance Covariance Methods

    Args:
        absolute_weights (np.ndarray): array with weights       
        df_returns (pd.DataFrame): dataframe with returns
        windows (list): list of windows for VaR estimation
        drop (bool): boolean to drop top 20 worst losses

    Returns:
        pd.DataFrame: dataframe with risk figures
    """

    df = pd.DataFrame(index=df_returns.index)
    df['Returns'] = np.sum(df_returns * absolute_weights, axis=1)

    if drop:
        to_drop = df.sort_values('Returns', ascending=False)[:20].index
        df = df.drop(to_drop)
        df_returns = df_returns.drop(to_drop)

    alphas = [0.975, 0.990]
    for alpha in alphas:
        for window in windows:

            means = np.empty(len(df_returns))
            stds = np.empty(len(df_returns))
            means[:] = np.NaN
            stds[:] = np.NaN
            for t in tqdm(range(window, len(df_returns))):
                means[t], stds[t] = var_covar(
                    df_returns[t-window:t], absolute_weights)
            df[f'VaR_VC_{window}_{alpha}_N'] = normal_var(means, stds, alpha)
            df[f'ES_VC_{window}_{alpha}_N'] = normal_es(means, stds, alpha)

            for dof in [3, 4, 5, 6]:
                df[f'VaR_VC_{window}_{alpha}_t{dof}'] = t_var(
                    dof, means, stds, alpha)
                df[f'ES_VC_{window}_{alpha}_t{dof}'] = t_es(
                    dof, means, stds, alpha)
    return df


# %%
windows = [250, 500, 1000]

df_var = VaR_VC(absolute_weights, df_losses, windows=windows, drop=False)
df_var_nonstressed = VaR_VC(
    absolute_weights, df_losses, windows=windows, drop=True)

# print(df_var_nonstressed.iloc[-1,:][df_var.columns.str.contains(f'(?=.*VaR)(?=.*N)',regex=True)].round(2).to_latex())
# df_var.to_csv('A1/Data/VaR/Variance-Covariance-Stressed.csv')
# df_var_nonstressed.to_csv('A1/Data/VaR/Variance-Covariance-Non-Stressed.csv')

# %%


def plotVaR(df: pd.DataFrame, title: str):
    """Plots certain columns from the generated VaR dataframe

    Args:
        df (pd.DataFrame): VaR dataframe for
        title (str): title for plot
    """
    df.dropna(inplace=True)
    plt.style.use("seaborn-muted")
    plt.figure(figsize=(12, 8))
    plt.plot(df['Returns'], color='lightgray')
    plt.plot(df[f'VaR_VC_{windows[0]}_0.975_t6'], color='red',
             linestyle='--', label=f'VaR_VC_{windows[0]}_0.975_t6')
    plt.plot(df[f'VaR_VC_{windows[0]}_0.975_t3'], color='green',
             linestyle='--', label=f'VaR_VC_{windows[0]}_0.975_t3')
    plt.plot(df[f'VaR_VC_{windows[0]}_0.975_N'], color='blue',
             linestyle='--', label=f'VaR_VC_{windows[0]}_0.975_N')
    plt.title(title)
    plt.ylim(0, 100000)
    plt.ylabel('Portfolio Losses (€)')
    plt.legend(loc=2)
    plt.show()

# %%
# Plot all calculated VaR's


def plotAllVaR(df: pd.DataFrame, title: str):
    df.dropna(inplace=True)
    plt.style.use("seaborn-muted")
    plt.figure(figsize=(12, 8))
    plt.plot(df['Returns'], color='lightgray')
    for column in df.columns[1:3]:
        plt.plot(df[f'{column}'],
                 linestyle='--', label=f'{column}')
    plt.title(title)
    plt.ylim(0, 75000)
    plt.ylabel('Portfolio Losses (€)')
    plt.legend()
    plt.show()


# %%
plotVaR(df_var, title='VaR including stressed periods')
plotVaR(df_var_nonstressed, title='VaR without worst 20 days')

# %%
plt.figure(figsize=(12, 8))
plt.plot(df_var['Returns'], color='lightgray')
plt.plot(df_var[f'VaR_VC_{windows[0]}_0.975_N'], color='red',
         linestyle='--', label=f'VaR_VC_{windows[0]}_0.975_N')
plt.plot(df_var_nonstressed[f'VaR_VC_{windows[0]}_0.975_N'], color='green',
         linestyle='--', label=f'VaR_VC_{windows[0]}_0.975_N')
plt.ylim(0)
plt.ylabel('Portfolio Losses (€)')
plt.legend()
plt.show()
# %%
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

ax1 = plt.subplot2grid(shape=(2, 6), loc=(0, 0), colspan=2)
ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=2)
ax3 = plt.subplot2grid((2, 6), (0, 4), colspan=2)
ax4 = plt.subplot2grid((2, 6), (1, 1), colspan=2)
ax5 = plt.subplot2grid((2, 6), (1, 3), colspan=2)

stats.probplot(df_var['Returns'], dist="norm", plot=ax1)
ax1.set_title("Normal Q-Q plot")

stats.probplot(df_var['Returns'], sparams=(3), dist=stats.t, plot=ax2)
ax2.set_title("Student-t (3) Q-Q plot")

stats.probplot(df_var['Returns'], sparams=(4), dist=stats.t, plot=ax3)
ax3.set_title("Student-t (4) Q-Q plot")

stats.probplot(df_var['Returns'], sparams=(5), dist=stats.t, plot=ax4)
ax4.set_title("Student-t (5) Q-Q plot")

stats.probplot(df_var['Returns'], sparams=(6), dist=stats.t, plot=ax5)
ax5.set_title("Student-t (6) Q-Q plot")

plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

plt.tight_layout()
plt.savefig('A1/Figures/QQ_plots.eps', format='eps')
plt.show()

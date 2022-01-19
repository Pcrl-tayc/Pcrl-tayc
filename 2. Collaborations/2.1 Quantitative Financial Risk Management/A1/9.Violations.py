# %%
# Data processing
import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import t as studentst
from scipy import stats
from pathlib import Path

# Graphing
from matplotlib import pyplot as plt

# Statistics
from scipy.stats import norm

from tqdm import tqdm

import os
# %%
dir = Path('A1/Data/VaR/')

df_dict= dict()
for file in dir.glob("*.csv"):
    # Getting the file name without extension
    file_name = os.path.splitext(os.path.basename(file))[0]
    # Reading the file content to create a DataFrame
    df_dict[file_name]= pd.read_csv(file, parse_dates=True, index_col=0)

# %%
key = list(df_dict.keys())[0]


df_violations = df_dict[key].copy()
df_violations = df_violations.iloc[:,:].lt(df_violations.iloc[:,0], axis=0)
for window in [250,500,1000]:
    print(f"VaR violations with {window}: \n {df_violations.sum()[df_violations.columns.str.contains(f'(?=.*VaR)(?=.*{window})',regex=True)].sort_values()}")
    print(f"ES violations with {window}: \n {df_violations.sum()[df_violations.columns.str.contains(f'(?=.*ES)(?=.*{window})',regex=True)].sort_values()}")

# with pd.option_context('display.max_rows', 
#                         None,
#                         'display.max_columns', 
#                         None):  # more options can be specified also
#     print(df_violations.sum().sort_values()[df_violations.sum()])
# %%

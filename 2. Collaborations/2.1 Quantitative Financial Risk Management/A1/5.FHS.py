#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
sns.set_style("darkgrid")
from tqdm import tqdm


# In[2]:


df_losses = pd.read_csv("Data\LinearLosses.csv",
                        parse_dates=True, index_col=0,) 

absolute_weights = np.array([500000, 500000, 500000, 500000, -1000000])


# In[5]:


def Hypot_EWMA(df_returns:pd.DataFrame,smooth:int)->pd.DataFrame:
   
    df_hyp=pd.DataFrame(index=df_returns.index,columns=df_returns.columns)    
                                                      
    
    
    EWMA=(df_returns.ewm(alpha=smooth,adjust=True)).std()
   
    std_inno=df_returns/EWMA
    df_hyp=(std_inno*EWMA)
    df_hyp=std_inno*(EWMA.shift(1))
    df_hyp.dropna(inplace=True)
    
    
    return df_hyp   

df_hyp=Hypot_EWMA(df_losses,0.03)


# In[6]:


# Defining the basic VaR function. 
def empirical_var(x, alpha):
    return np.quantile(np.sort(x), alpha, interpolation='linear')

# Defining the basic ES function. 

def empirical_es(x, alpha):
    var = empirical_var(x, alpha)
    return x[x >= var].mean()


def VaR_HS(df_returns:pd.DataFrame, windows:list, absolute_weights:np.ndarray)->pd.DataFrame:
    df=pd.DataFrame(index=df_returns.index)
    df["Returns"]=np.sum(df_returns*absolute_weights, axis=1)
    alphas=[0.975,0.99]
    
    for alpha in alphas:
        
        for window in windows:            
            df[f'VaR_{window}_{alpha}_FHS']=np.NaN
            df[f'ES_{window}_{alpha}_FHS']=np.NaN
            
            for t in tqdm (range(window,len(df_returns))):
                df[f'VaR_{window}_{alpha}_FHS'].iloc[t]=empirical_var(df['Returns'].iloc[t-window:t], alpha)
                df[f'ES_{window}_{alpha}_FHS'].iloc[t] = empirical_es(df['Returns'].iloc[t-window:t], alpha)
    return df


# In[7]:


windows = [300]
df_var = VaR_HS(df_hyp,windows,absolute_weights)


# In[8]:


df_var.plot(figsize=(15,9),title="VaR_FHS",ylim=(0),ylabel="Losses",xlabel="Period [days]",legend="reverse");


# ### AMIEL´S

# In[12]:


def FHS_EWMA(absolute_weights: np.ndarray, df_returns:pd.DataFrame,smooth:int, windows:list)->pd.DataFrame:
   
    #Creating the dataframe of innovations
    df_hyp=pd.DataFrame(index=df_returns.index,columns=df_returns.columns)                                                         
    #Computing the EWMA of the dataframe passed as input
    df_EWMA=df_returns.ewm(alpha=smooth,adjust=False).std() 
    #Computing the innovations for each risk-factor                      
    std_inno=df_returns/df_EWMA
    df_hyp=std_inno*(EWMA.shift(1))
    df_hyp.dropna(inplace=True)
                                                 
    
    df_VaR = VaR_HS(absolute_weights, df_hyp, windows)
    df_VaR.columns = ['VaR_FHS_300_0.975_N','ES_FHS_300_0.975_N','VaR_FHS_300_0.99_N','ES_FHS_300_0.99_N']
    return df_VaR   


# In[13]:


windows = [300, 1250, 2000]
df_var = VaR_HS(df_losses,windows,absolute_weights)


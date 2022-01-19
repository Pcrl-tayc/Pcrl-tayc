import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap(df, higher_is_better=True, ylabel='L1 ratio', xlabel='alpha'):
    """ Plots 2-dimensional gridsearch results via heatmap and marks the optimal values
    
    Parameters
    ----------
    df : pd.DataFrame (n_l1ratios, n_alphas)
        DataFrame with index that contains param1 and columns with param2 hyperparameters. 
        DataFrame contains scoring results.
        
    higher_is_better : bool, optional
        Set to True (default) if estimator is optimized by maximizing a value,
        such as r2 or neg_mean_squared_error.
        See '3.3.1. The scoring parameter' on sklearn for more details
        
    ylabel : str, optional
        String for y-axis label
        
    xlabel : str, optional
        String for x-axis label
        
    Returns
    -------
    fig : matplotlib Figure object
        Returns the figure object that can be further manipulated outside
        of this function.
        
    Example
    -------
    print(df) yields
    
    param1        -4.95     -4.68     -4.42 ...
    param2                              
    0.6      -0.013721 -0.013590 -0.013437 
    0.8      -0.013668 -0.013520 -0.013341 ...
    1.0      -0.013619 -0.013466 -0.013272 
    ...                   ...
    
    """

    fig, ax = plt.subplots(1,1)
    
    cmap = plt.get_cmap("Blues_r")
    sns.heatmap(df, ax=ax, cmap=cmap)
    
    if not higher_is_better:
        df = -df
        
    xvals = []
    yvals = []
    for yval,xval in df.idxmax(axis=1).iteritems():
        j = (df.index == yval).argmax() + .5
        i = (df.columns == xval).argmax() + .5
        xvals.append(j)
        yvals.append(i)
        
    ax.plot(yvals, xvals, marker='o', color='red')
    
    xoptm = df.idxmax(axis=1).max()
    print(df.loc[:,xoptm])
    yoptm = df.loc[:,xoptm].idxmax()
    j = (df.index == yoptm).argmax() + .5
    i = (df.columns == xoptm).argmax() + .5
    ax.plot(i, j, marker='s', markersize=10, color='red')
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(f'GridSearch results\noptimal {ylabel}={yoptm:.1f}, {xlabel}={xoptm:.1f}')

    return fig
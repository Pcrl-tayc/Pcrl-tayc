import pandas as pd
import numpy as np
        
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import seaborn as sns

cmap2 = ListedColormap(['royalblue', 'orangered'])
cmap3 = ListedColormap(['royalblue', 'seagreen', 'orangered'])
        
        

def plot_confusion_matrix(y_true, y_pred, labels,
                          normalize=False, title=None, cmap=plt.cm.Blues):
    """ Draws a confusion matrix
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Target data
        
    y_pred : array-like of shape (n_samples,)
        Predicted target data
        
    labels : array-like 
        List of strings that describe the different types
        of target data
        
    normalize : bool, optional
        If true, the rows of the confusion matrix add up to 1.
        
    title : str, optional
        Sets the title for the plotted matrix
        
    cmap : matplotlib colormap
        Sets the color scale used in the plotted matrix
    
    """
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Only use the labels that appear in the data
    labels = np.array(labels)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    # Plot confusion matrix as heatmap
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=labels,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax



def plot_classification_contours(X, y, clf, labels, h=0.02):
    """ Plots non-linear decision regions for a classifier
    
    Parameters
    ----------
    X : {array-like} of shape (n_samples, n_features)
        Training data. 
        
    y : array-like of shape (n_samples,)
        Target data
        
    clf : object
        A sklearn classifier that has the .predict() method
        
    h : float, optional
        Determines how fine the background grid is, upon
        which the classifier is called.
    
    """
    
    cm = cmap2 if len(labels) == 2 else cmap3

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(1, 1)

    ax.contourf(xx, yy, Z, cmap=cm, alpha=.3)
    ax.scatter(X[y==y.min(), 0], X[y==y.min(), 1], color=cm(0), cmap=cm, edgecolors='k')
    ax.scatter(X[y!=y.min(), 0], X[y!=y.min(), 1], color=cm(1), cmap=cm, edgecolors='k')

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    
    fig.tight_layout()
    
    return fig


def plot_pairwise_scatter(sr_targets, df_features, labels):
    """ Produces a scatter matrix, colored by target values
    
    Results in a lower-triangular matrix where all features are scattered
    against each other. The coloring depends on what value of the target
        
    sr_targets : pandas.Series of shape (n_samples,)
        Target data
    
    df_features : pandas.DataFrame of shape (n_samples, n_features)
        Training data. 
        
    labels : list-like, optional
        In ascending order, the labels for the values in sr_targets
    
    """
    
    nb = df_features.shape[1]
    fig, axes = plt.subplots(nb, nb, figsize=[10,10])
    
    label_values = list(sorted(sr_targets.unique()))
    if len(label_values) == 2:
        targets = (sr_targets == label_values[0]).astype(int).values
    else:
        targets = sr_targets.values
    targets_unique = list(set(targets))
    
    cm = cmap2 if len(targets_unique)==2 else cmap3
    
    for i, (f1, d1) in enumerate(df_features.iteritems()):
        for j, (f2, d2) in enumerate(df_features.iteritems()):
            ax = axes[i,j]
            
            if i<j: 
                ax.axis('off')
                continue

            if i==j:
                for t, target in enumerate(targets_unique):
                    d = d1[targets==target].values
                    sns.kdeplot(d, ax=ax, shade=True, color=cm(t))
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f1)
            else:
                ax.scatter(d1, d2, c=targets, cmap=cm, s=3)
                if j>0: ax.set_yticks([])
                if i<nb-1: ax.set_xticks([])

    fig.tight_layout()
    
    ax = axes[0,0]
    ax.legend(labels, loc='upper left', bbox_to_anchor=[1,1])
    
    
    
get_ipython().run_line_magic('matplotlib', 'inline')

#------------------------------------------------------------------------------------------------------------------------------
# By Alexandra Lee 
# (updated October 2018) 
#
# Plot correlation between reconstructed vs observed gene expression per sample 
#-------------------------------------------------------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

randomState = 123
from numpy.random import seed
seed(randomState)

def plot_corr_gradient(out_dir, viz_dir):
    """
    For each sample plot correlation score estimated after performing interporlation in gene space and latent space
    Plot use LOWESS smoothing to identify overall trend in the data
    
    """
    
    # load arguments
    gene_space_file = os.path.join(out_dir, "corr_gene_space.txt")
    latent_space_file = os.path.join(out_dir, "corr_latent_space.txt")
    sorted_target_gene_file = os.path.join(out_dir, "sorted_id.txt")

    # output
    fig1_file = os.path.join(viz_dir, "smooth.png")
    fig2_file = os.path.join(viz_dir, "corr.png")

    # read in data
    gene_corr_data = pd.read_table(gene_space_file, header=0, sep='\t', index_col=0, names=['gene_space'])
    latent_corr_data = pd.read_table(latent_space_file, header=0, sep='\t', index_col=0, names=['latent_space'])
    sorted_id = pd.read_table(sorted_target_gene_file, header=0, sep='\t', index_col=0)

    # Join 
    X = pd.merge(gene_corr_data, latent_corr_data, left_index=True, right_index=True)
    X.head(5)


    # Plot smoothing curve 
    num_samples = gene_corr_data.shape[0]
    x = np.linspace(1,num_samples, num=num_samples)
    X_sorted = X.loc[sorted_id.index].dropna()

    fg = sns.regplot(x=x, y='gene_space', data=X_sorted, lowess=True, scatter=False, color="orchid", label="Gene space correlation")
    fg = sns.regplot(x=x, y='latent_space', data=X_sorted, lowess=True, scatter=False, color="darkturquoise", label="Latent space correlation")
    ax = fg.axes
    ax.set_xlim(0,num_samples)
    ax.set_ylim(0.4,0.9)
    fg.set_ylabel('Pearson correlation score')
    fg.set_xlabel('Sorted samples')
    fg.legend()
    fig = fg.get_figure()
    fig.savefig(fig1_file, dpi=300)


    # Plot correlation of gene space vs latent space
    # Plot
    fg=sns.jointplot(x='gene_space', y='latent_space', data=X, kind='hex');
    fg.set_axis_labels('Gene space correlation', 'Latent space correlation')
    fg.savefig(fig2_file)


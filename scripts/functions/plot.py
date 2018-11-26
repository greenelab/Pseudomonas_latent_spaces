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
from statsmodels.nonparametric.smoothers_lowess import lowess

randomState = 123
from numpy.random import seed
seed(randomState)

def plot_corr_gradient(out_dir, viz_dir):
    """
    For each sample plot correlation score estimated after performing interporlation in gene space and different latent spaces
    Plot use LOWESS smoothing to identify overall trend in the data
    
    """
    
    # load arguments
    gene_space_file = os.path.join(out_dir, "corr_gene_space.txt")
    latent_space_vae_file = os.path.join(out_dir, "corr_latent_space_vae.txt")
    latent_space_pca_file = os.path.join(out_dir, "corr_latent_space_pca.txt")
    sorted_target_gene_file = os.path.join(out_dir, "sorted_id.txt")

    # output
    fig1_file = os.path.join(viz_dir, "smooth_redo.png")
    fig2_file = os.path.join(viz_dir, "corr_gene_vs_vae_redo.png")
    fig3_file = os.path.join(viz_dir, "corr_gene_vs_pca_redo.png")

    # read in data
    gene_corr_data = pd.read_table(gene_space_file, header=0, index_col=0, names=['gene_space'])
    latent_vae_corr_data = pd.read_table(latent_space_vae_file, header=0, index_col=0, names=['latent_space_vae'])
    latent_pca_corr_data = pd.read_table(latent_space_pca_file, header=0, index_col=0, names=['latent_space_pca'])
    sorted_id = pd.read_table(sorted_target_gene_file, header=0, index_col=0, names=['gene_A_expression'])

    # Join 
    X_tmp = pd.merge(gene_corr_data, latent_vae_corr_data, left_index=True, right_index=True)
    #X = pd.merge(X_tmp, latent_pca_corr_data, left_index=True, right_index=True)
    X_tmp2 = pd.merge(X_tmp, latent_pca_corr_data, left_index=True, right_index=True)
    X = pd.merge(X_tmp2, sorted_id, left_index=True, right_index=True)
    
    # Plot smoothing curve 
    num_samples = gene_corr_data.shape[0]
    #x = np.linspace(1,num_samples, num=num_samples)
    X_sorted = X.loc[sorted_id.index].dropna()

    #print(X_sorted)
    #fg = sns.regplot(x=x, y='gene_space', data=X_sorted, lowess=True, scatter=False, color="mediumorchid", label="Gene space correlation")
    #fg = sns.regplot(x=x, y='latent_space_vae', data=X_sorted, lowess=True, scatter=False, color="teal", label="Latent space VAE correlation")
    #fg = sns.regplot(x=x, y='latent_space_pca', data=X_sorted, lowess=True, scatter=False, color="indigo", label="Latent space PCA correlation")
    fg = sns.regplot(x='gene_A_expression', y='gene_space', data=X_sorted, lowess=True, scatter=False, color="mediumorchid", label="Gene space correlation")
    fg = sns.regplot(x='gene_A_expression', y='latent_space_vae', data=X_sorted, lowess=True, scatter=False, color="teal", label="Latent space VAE correlation")
    fg = sns.regplot(x='gene_A_expression', y='latent_space_pca', data=X_sorted, lowess=True, scatter=False, color="indigo", label="Latent space PCA correlation")
    ax = fg.axes
    #ax.set_xlim(0,num_samples)
    ax.set_xlim(0,1.0)
    ax.set_ylim(0.0,1.0)
    fg.set_ylabel('Pearson correlation score')
    #fg.set_xlabel('Sorted samples')
    fg.set_xlabel('Expression of gene A')
    fg.legend()
    fig = fg.get_figure()
    fig.savefig(fig1_file, dpi=300)


    # Plot correlation of gene space vs latent space
    fg=sns.jointplot(x='gene_space', y='latent_space_vae', data=X, kind='hex');
    fg.set_axis_labels('Gene space correlation', 'Latent space VAE correlation')
    #fg.set_title('Gene space vs Latent VAE space')
    fg.savefig(fig2_file)
    
    #fg=sns.jointplot(x='gene_space', y='latent_space_pca', data=X, kind='hex');
    fg.set_axis_labels('Gene space correlation', 'Latent space PCA correlation')
    #fg.set_title('Gene space vs Latent PCA space')
    fg.savefig(fig3_file)

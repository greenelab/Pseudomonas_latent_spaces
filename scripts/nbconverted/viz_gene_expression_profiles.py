
# coding: utf-8

# # Visualize
# Gene expression data in raw gene space vs low dimensional latent space using UMAP

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import pandas as pd
import numpy as np
import random
import glob
import umap
import seaborn as sns
import matplotlib.pyplot as plt

from numpy.random import seed
randomState = 123
seed(randomState)


# In[2]:


# Load 
base_dir = os.path.dirname(os.getcwd())
analysis_name = 'sim_balancedAB_2latent'

sim_data_file = os.path.join(
    base_dir,
    "data",
    analysis_name,
    "train_model_input.txt.xz"
)
data_encoded_file = glob.glob(os.path.join(
    base_dir,
    "encoded",
    analysis_name,
    "train_input_2layer_2latent_encoded.txt"))[0]
A_file = os.path.join(
    base_dir,
    "data",
    analysis_name,
    "geneSetA.txt"
)


# In[3]:


# Read data
sim_data = pd.read_table(sim_data_file, index_col=0, header=0, compression='xz')
sim_data_encoded = pd.read_table(data_encoded_file, header=0, index_col=0)
geneSetA = pd.read_table(A_file, header=0, index_col=0)


# In[4]:


sim_data.head()


# In[5]:


sim_data_encoded.head()


# In[6]:


geneSetA.head()


# In[7]:


# Label samples by gene A expression

# Since our simulation set all genes in set A to be the same value for a give sample
# we can consider a single gene in set A to query by
rep_gene_A = geneSetA.iloc[0][0]
geneA_exp = sim_data[rep_gene_A]

sample_id = sim_data.index

# Bin gene A expression
geneA_exp_labeled = sim_data.assign(
    rep_geneA=(
        list( 
            map(
                lambda x:
                '1' if 0 < x and x <=0.1 
                else '2' if 0.1< x and x <=0.2 
                else '3' if 0.2<x and x<=0.3
                else '4' if 0.3<x  and x<=0.4
                else '5' if 0.4<x and x<=0.5
                else '6' if 0.5<x and x<=0.6
                else '7' if 0.6<x and x<=0.7
                else '8' if 0.7<x and x<=0.8
                else '9' if 0.8<x and x<=0.9
                else '10',
                geneA_exp
            )      
        )
    )
)
geneA_exp_labeled = geneA_exp_labeled.astype({"rep_geneA": int})
geneA_exp_labeled.head()


# ## Plot gene expression in gene space
# 
# Each dot is a sample.  Each sample is colored based on its expression of gene A
# 
# In the legend 1 ~ gene A expression is (0.0, 0.1], 2 ~ gene A expression is (0.1, 0.2], etc.

# In[8]:


# UMAP embedding of raw gene space data
embedding = umap.UMAP().fit_transform(sim_data)
embedding.shape


# In[9]:


# UMAP plot of raw gene expression data
geneA_exp_labeled = geneA_exp_labeled.assign(sample_index=list(range(geneA_exp_labeled.shape[0])))
for x in geneA_exp_labeled.rep_geneA.sort_values().unique():
    plt.scatter(
        embedding[geneA_exp_labeled.query("rep_geneA == @x").sample_index.values, 0], 
        embedding[geneA_exp_labeled.query("rep_geneA == @x").sample_index.values, 1], 
        c=sns.color_palette()[x-1],
        alpha=0.7,
        label=str(x)
    )
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of gene expression data in GENE space', fontsize=14)
plt.legend()


# ## Plot gene expression in VAE encoded latent space

# In[10]:


# Plot of gene expression data in VAE latent space
geneA_exp_labeled = geneA_exp_labeled.assign(sample_index=list(range(geneA_exp_labeled.shape[0])))
for x in geneA_exp_labeled.rep_geneA.sort_values().unique():
    sample_ids = list(geneA_exp_labeled.query("rep_geneA == @x").sample_index.index)
    plt.scatter(
        sim_data_encoded.loc[sample_ids, '1'], 
        sim_data_encoded.loc[sample_ids, '2'], 
        c=sns.color_palette()[x-1],
        alpha=0.7,
        label=str(x)
    )
plt.gca().set_aspect('equal', 'datalim')
plt.title('Encoded gene expression data in VAE latent space', fontsize=14)
plt.legend()


# In[11]:


# UMAP embedding of VAE encoded gene space data
embedding_encoded = umap.UMAP().fit_transform(sim_data_encoded)
embedding_encoded.shape


# In[12]:


# UMAP plot of raw gene expression data
geneA_exp_labeled = geneA_exp_labeled.assign(sample_index=list(range(geneA_exp_labeled.shape[0])))
for x in geneA_exp_labeled.rep_geneA.sort_values().unique():
    plt.scatter(
        embedding_encoded[geneA_exp_labeled.query("rep_geneA == @x").sample_index.values, 0], 
        embedding_encoded[geneA_exp_labeled.query("rep_geneA == @x").sample_index.values, 1], 
        c=sns.color_palette()[x-1],
        alpha=0.3,
        label=str(x)
    )
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of encoded gene expression data in VAE space', fontsize=14)
plt.legend()


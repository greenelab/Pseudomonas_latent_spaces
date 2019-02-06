
# coding: utf-8

# # Explore simulated data
# 
# We explore the input data used to train the VAE.  
# 
# Is our artifical signal that we have added to the data obvious enough that we expect the VAE to detect it?

# In[ ]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels.nonparametric.smoothers_lowess import lowess
import math as mt

randomState = 123
from numpy.random import seed
seed(randomState)


# In[2]:


# Load data
base_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
simulation_name = "sim_AB_2775_300_v2"
data_file = os.path.join(base_dir, simulation_name, "train_model_input.txt.xz")
A_file = os.path.join(base_dir, simulation_name, "geneSetA.txt")
B_file = os.path.join(base_dir, simulation_name, "geneSetB.txt")


# In[3]:


# Read data
data = pd.read_table(data_file, header=0, index_col=0, compression='xz')
data.head()


# ## 1. Distribution of gene A across samples

# In[4]:


gene_id = 'PA3423' ## search in files for "PA" pattern
A_only = pd.DataFrame(data[gene_id], index=data.index, columns=[gene_id])
sns.distplot(A_only)


# ## 2. Are the genes changing linearly with respect to A?

# In[5]:


# Sort samples by gene A expression (lowest --> highest)
data_sorted = data.sort_values(by=[gene_id])
data_sorted.head()


# In[6]:


# For each gene plot gene expression trend along A gradient
# Use pearson correlation score to compare A profile with all other genes
# Pearson correlation evaluates the linear relationship between two continuous variables
data_corr = data_sorted.corr(method='pearson')
data_corr.head()


# In[7]:


# Plot distribution of correlation scores
data_corr_A = data_corr[gene_id]
sns.distplot(data_corr_A)


# ## 3.  What does the data look like
# Heatmap of genes clustered by expression pattern along gene A gradient

# In[8]:


# Expect to see clustering of genes in group B that increase when gene A > 0.5
# Expect to see clustering of genes in group A with identical expression
# Keep track if gene is in group A or B
# To be fed into explore_simulated_data.py
geneSetA = pd.read_table(A_file, header=0, index_col=0)
geneSetB = pd.read_table(B_file, header=0, index_col=0)

geneSetA = geneSetA['0'].values.tolist()
geneSetB = geneSetB['0'].values.tolist()


# In[9]:


# Sort samples by gene A expression (lowest --> highest)
data_sorted = data.sort_values(by=[gene_id])
data_sorted.head()


# In[10]:


# Add group labels per gene (group A, B, None)
genes = data.T.index

data_sorted_labeled = data_sorted.T.assign(
    gene_group=(
        list( 
            map(
                lambda x: 'A' if x in geneSetA else 'B' if x in geneSetB else 'None',
                genes
            )      
        )
    )
)
data_sorted_labeled.head()


# In[11]:


# Heatmap sorted by gene expression signature
# colormap: 
#     A - green
#     B - blue
#     None - red
gene_groups = data_sorted_labeled["gene_group"]
lut = dict(zip(gene_groups.unique(), "rbg"))
row_colors = gene_groups.map(lut)

sns.clustermap(data_sorted.T,
               row_cluster=True,
               col_cluster=False,
               metric="correlation",
               row_colors=row_colors,
               figsize=(20,10))


# ## 4.  Plot the expression of genes along the gene A gradient
# Are the genes changing linearly with respect to gene A?

# In[12]:


get_ipython().run_cell_magic('time', '', '# Sort samples by PA1673 expression (lowest --> highest)\ndata_sorted = data.sort_values(by=[gene_id])\ndata_sorted.head()\n\n# For each gene plot gene expression trend along Pgene A gradient\n\n# Initialize the figure\nplt.style.use(\'seaborn-darkgrid\')\n \n# create a color palette\npalette = plt.get_cmap(\'Set1\')\n \n# multiple line plot\nnum_genes = data_sorted.shape[1]\nnum_panels = 100\n\n# Output file directory\nbase_dir = os.path.join(os.path.dirname(os.getcwd()), "exploration_results", simulation_name)\nos.makedirs(base_dir, exist_ok=True)\n    \nfor panel in range(mt.ceil(num_genes%num_panels)):\n    rows = mt.sqrt(num_panels)\n    cols = rows\n    num=0\n    \n    # Not incrementing correctly here\n    data_subsample = data_sorted.drop(gene_id, axis=1).iloc[: , (panel*num_panels):((panel+1)*num_panels)-1]\n    \n    for column in data_subsample:\n        num+=1\n\n        # Find the right spot on the plot\n        plt.subplot(rows,cols, num)\n\n        # Plot the lineplot --Add smoothing to see trend\n        y_smooth = lowess(data_sorted[gene_id], data_subsample[column])[:,1]\n        plt.plot(data_sorted[gene_id], y_smooth, marker=\'\', color=palette(num), linewidth=1.9, alpha=0.9, label=column)\n               \n        # Same limits for everybody!\n        plt.xlim(0,1)\n        plt.ylim(0,1)\n\n        plt.tick_params(labelbottom=\'off\')\n        plt.tick_params(labelleft=\'off\')\n        # Not ticks everywhere\n        #if num in range(7) :\n        #    plt.tick_params(labelbottom=\'off\')\n        #if num not in [1,4,7] :\n        #    plt.tick_params(labelleft=\'off\')\n\n        # Add title\n        plt.title(column, loc=\'left\', fontsize=5, fontweight=0, color=palette(num) )\n\n    # general title\n    plt.suptitle("How gene expression changed\\nalong gene {} gradient?".format(gene_id), fontsize=13, fontweight=0, color=\'black\', style=\'italic\', y=1.02)\n\n    # Axis title\n    plt.text(0.5, 0.02, \'PA1673 expression\', ha=\'center\', va=\'center\')\n    plt.text(0.06, 0.5, \'Gene i expression\', ha=\'center\', va=\'center\', rotation=\'vertical\')\n    \n    # Save each panel as a figure\n    file_out = PdfPages(os.path.join(base_dir, \'Panel_{}.pdf\'.format(panel)))\n    plt.savefig(file_out, format=\'pdf\', bbox_inches = \'tight\')\n    plt.show()\n    file_out.close()')


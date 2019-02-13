
# coding: utf-8

# # Explore RNA-seq input dataset
# 
# Look into patterns in the RNA-seq dataset that is used as input into the interpolation analysis performed.
# Is there a way to explain the results we are seeing?

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
base_dir = os.path.join(os.path.dirname(os.getcwd()), 'data')
data_file = os.path.join(base_dir, "all-pseudomonas-gene-normalized.zip")


# In[3]:


# Read data
data = pd.read_table(data_file, header=0, index_col=0, compression='zip')
data = data.T
data.head()


# ## 1. Distribution of PA1673 across samples
# 
# We ultimately want to capture gene expression changes that are related to changes in oxygen concentration.  Since we know that PA1673 is regulated by oxygen concenration, we use the expression of this gene as a "phenotype" and look for correlations in gene expression changes and changes in PA1673 expression.

# In[4]:


gene_id = 'PA1673'
PA1673_only = pd.DataFrame(data[gene_id], index=data.index, columns=[gene_id])
sns.distplot(PA1673_only)


# ## 2. Are the genes changing linearly with respect to PA1673?

# In[5]:


# Sort samples by PA1673 expression (lowest --> highest)
data_sorted = data.sort_values(by=[gene_id])
data_sorted.head()


# In[6]:


# For each gene plot gene expression trend along PA1673 gradient
# Use pearson correlation score to compare PA1673 profile with all other genes
# Pearson correlation evaluates the linear relationship between two continuous variables
data_corr = data_sorted.corr(method='pearson')
data_corr.head()


# In[7]:


# Plot distribution of correlation scores
data_corr_PA1673 = data_corr[gene_id]
sns.distplot(data_corr_PA1673)


# ## 3.  Plot the expression of genes along the PA1673 gradient
# Are the genes changing linearly with respect to PA1673?

# In[8]:


get_ipython().run_cell_magic('time', '', '# Sort samples by PA1673 expression (lowest --> highest)\ndata_sorted = data.sort_values(by=[gene_id])\ndata_sorted.head()\n\n# For each gene plot gene expression trend along PA1673 gradient\n\n# Initialize the figure\nplt.style.use(\'seaborn-darkgrid\')\n \n# create a color palette\npalette = plt.get_cmap(\'Set1\')\n \n# multiple line plot\nnum_genes = data_sorted.shape[1]\nnum_panels = 100\n\n# Output file directory\nbase_dir = os.path.join(os.path.dirname(os.getcwd()), "exploration_results", "real_data")\nos.makedirs(base_dir, exist_ok=True)\n    \nfor panel in range(mt.ceil(num_genes%num_panels)):\n    rows = mt.sqrt(num_panels)\n    cols = rows\n    num=0\n    \n    data_subsample = data_sorted.drop(gene_id, axis=1).iloc[: , (panel*num_panels):((panel+1)*num_panels)-1]\n    \n    for column in data_subsample:\n        num+=1\n\n        # Find the right spot on the plot\n        plt.subplot(rows,cols, num)\n\n        # Plot the lineplot --Add smoothing to see trend\n        y_smooth = lowess(data_sorted[gene_id], data_subsample[column])[:,1]\n        plt.plot(data_sorted[gene_id], y_smooth, marker=\'\', color=palette(num), linewidth=1.9, alpha=0.9, label=column)\n               \n        # Same limits for everybody!\n        plt.xlim(0,1)\n        plt.ylim(0,1)\n\n        plt.tick_params(labelbottom=\'off\')\n        plt.tick_params(labelleft=\'off\')\n        # Not ticks everywhere\n        #if num in range(7) :\n        #    plt.tick_params(labelbottom=\'off\')\n        #if num not in [1,4,7] :\n        #    plt.tick_params(labelleft=\'off\')\n\n        # Add title\n        plt.title(column, loc=\'left\', fontsize=5, fontweight=0, color=palette(num) )\n\n    # general title\n    plt.suptitle("How gene expression changed\\nalong gene {} gradient?".format(gene_id), fontsize=13, fontweight=0, color=\'black\', style=\'italic\', y=1.02)\n\n    # Axis title\n    plt.text(0.5, 0.02, \'Gene {} expression\'.format(gene_id), ha=\'center\', va=\'center\')\n    plt.text(0.06, 0.5, \'Gene i expression\', ha=\'center\', va=\'center\', rotation=\'vertical\')\n    \n    # Save each panel as a figure\n    file_out = PdfPages(os.path.join(base_dir, \'Panel_{}.pdf\'.format(panel)))\n    plt.savefig(file_out, format=\'pdf\', bbox_inches = \'tight\')\n    plt.show()\n    file_out.close()')



# coding: utf-8

# In[1]:


#-------------------------------------------------------------------------------------------------------------------------------
# By Alexandra Lee
# (created January 2019)
# 
# Explore Pseudomonas dataset in order to explain interpolation results
#-------------------------------------------------------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math as mt

randomState = 123
from numpy.random import seed
seed(randomState)


# In[2]:


# Load data
base_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
simulation_name = "sim_AB_2775_300_v2"
data_file = os.path.join(base_dir, simulation_name, "train_model_input.txt.xz")


# In[3]:


# Read data
data = pd.read_table(data_file, header=0, index_col=0, compression='xz')
data.head()


# In[4]:


## 1. Distribution of gene A across samples

gene_id = 'PA3423' ## search in files for "PA" pattern
A_only = pd.DataFrame(data[gene_id], index=data.index, columns=[gene_id])
sns.distplot(A_only)


# In[5]:


## 2. Are the genes changing linearly with respect to A?

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


# In[17]:


## 3.  Heatmap of genes clustered by expression pattern along gene A gradient
# Expect to see clustering of genes in group B that increase when gene A > 0.5
# Expect to see clustering of genes in group A with identical expression
sns.clustermap(data_sorted.T,
               row_cluster=True,
               col_cluster=False,
               metric="correlation",
               figsize=(20,10))


# In[8]:


# %%time
# # Plot the expression of genes along the PA1673 gradient:
# # Are the genes changing linearly with respect to PA1673?

# # Sort samples by PA1673 expression (lowest --> highest)
# data_sorted = data.sort_values(by=[gene_id])
# data_sorted.head()

# # For each gene plot gene expression trend along PA1673 gradient

# # Initialize the figure
# plt.style.use('seaborn-darkgrid')
 
# # create a color palette
# palette = plt.get_cmap('Set1')
 
# # multiple line plot
# num_samples = data_sorted.shape[0]
# num_panels = 100

# for panel in range(num_samples%num_panels):
#     rows = mt.sqrt(num_panels)
#     cols = rows
#     num=0
    
#     # Not incrementing correctly here
#     data_subsample = data_sorted.drop(gene_id, axis=1).iloc[:,panel:num_panels]
    
#     for column in data_subsample:
#         num+=1

#         # Find the right spot on the plot
#         plt.subplot(rows,cols, num)

#         # Plot the lineplot
#         plt.plot(data_sorted[gene_id], data_subsample[column], marker='', color=palette(num), linewidth=1.9, alpha=0.9, label=column)

#         # Same limits for everybody!
#         plt.xlim(0,1)
#         plt.ylim(0,1)

#         # Not ticks everywhere
#         #if num in range(7) :
#         #    plt.tick_params(labelbottom='off')
#         #if num not in [1,4,7] :
#         #    plt.tick_params(labelleft='off')

#         # Add title
#         plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(num) )

#     # general title
#     plt.suptitle("How gene expression changed\nalong PA1673 gradient?", fontsize=13, fontweight=0, color='black', style='italic', y=1.02)

#     # Axis title
#     plt.text(0.5, 0.02, 'PA1673 expression', ha='center', va='center')
#     plt.text(0.06, 0.5, 'Gene i expression', ha='center', va='center', rotation='vertical')
    
#     # Save each panel as a figure
#     base_dir = os.path.join(os.path.dirname(os.getcwd()), 'VAE', 'output')
#     file_out = PdfPages(os.path.join(base_dir, 'Panel_{}.pdf'.format(panel)))
#     plt.savefig(file_out, format='pdf', bbox_inches = 'tight')
#     plt.show()
#     file_out.close()


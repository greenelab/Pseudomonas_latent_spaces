
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#-------------------------------------------------------------------------------------------------------------------------------
# By Alexandra Lee (July 2018) 
#
# Visualize Pseudomonas gene expression data projected onto t-SNE dimensions
# 
# Input: Pa gene expression data from ArrayExpress (matrix: sample x gene)
# Data compression method: None
# Output: Original Pa gene expression data projected onto t-SNE dimensions 
#-------------------------------------------------------------------------------------------------------------------------------
import os
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
np.random.seed(123)


# In[3]:


# load arguments
data_file = os.path.join(os.path.dirname(os.getcwd()), "data", "all-pseudomonas-gene-normalized.pcl")
map_file = os.path.join(os.path.dirname(os.getcwd()), "metadata", "mapping_sampleID_medium.txt")


# In[4]:


# read in data
data = pd.read_table(data_file, header=0, sep='\t', index_col=0)
X = data.transpose()
X.head(5)


# In[5]:


# Plot distribution of genes
# Values are normalized log2 transformed gene expression per gene (i.e. capture differential expression per gene)
#plt.hist(X['PA5570'])


# In[6]:


# tSNE
tsne = TSNE(n_components=2, init='pca', random_state=123, perplexity=30, learning_rate=300, n_iter=400)
tsne_X = tsne.fit_transform(X)
tsne_X


# In[7]:


# Map sample id to clinial phenotype (i.e. experimental condition)

# Note:
# According to the source (https://github.com/scikit-learn/scikit-learn/blob/14031f6/sklearn/decomposition/pca.py#L310),
# input will be transformed by np.array() before doing PCA. So row index will be lost during 
# PCA.fit_transform(X) even using a structured array or a pandas DataFrame. However, the order of the data is preserved, 
# meaning you can attach the index back afterward
X_ann = pd.DataFrame(tsne_X, index=X.index, columns=['tsne1', 'tsne2'])

# read in mapping file (sample id -- phenotype)
mapper = pd.read_table(map_file, header=0, sep='\t', index_col=0)

# Join 
X_new = pd.merge(X_ann, mapper, left_index=True, right_index=True)
X_new.head(5)


# In[8]:


# Plot
# Note: t-SNE has a cost function that is not convex, i.e. with different initializations we can get different results.
fg = sns.lmplot(x='tsne1', y='tsne2', data=X_new, hue='medium', fit_reg=False)
fg.add_legend()
fig_file = os.path.join(os.path.dirname(os.getcwd()), "viz", "all_dim.png")
fg.fig.suptitle('No compression')
fg.savefig(fig_file)

#fg = sns.FacetGrid(data=X_new, hue='medium', aspect=1.61)
#fg.map(plt.scatter, 'tsne1', 'tsne2')
#fg.add_legend()
#legend_file = 'C:/Users/alexj/Documents/UPenn/CGreene/Pseudomonas/output/all_dim_legend.png'
#fig_file = 'C:/Users/alexj/Documents/UPenn/CGreene/Pseudomonas/output/all_dim.png'
#fg.savefig(fig_file)
#fg.savefig(legend_file)


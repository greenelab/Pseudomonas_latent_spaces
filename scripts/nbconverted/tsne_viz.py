
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#-------------------------------------------------------------------------------------------------------------------------------
# By Alexandra Lee (July 2018) 
#
# Visualize Pseudomonas encoded gene expression data using t-SNE
# Encoding performed using Tybalt (VAE) or eADAGE (DA)
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
file_name = 'tybalt_2layer_encoded_10.tsv'
encoded_data_file = os.path.join(os.path.dirname(os.getcwd()), "encoded", file_name)
map_file = os.path.join(os.path.dirname(os.getcwd()), "metadata", "mapping_sampleID_medium.txt")


# In[4]:


# read in encoded data
X = pd.read_table(encoded_data_file, header=0, sep='\t', index_col=0)
#X = pd.read_table(encoded_data_file, header = None, sep = '\t', index_col = None) # eADAGE doesn't have header
X = pd.DataFrame(X)
X.head(5)
#X.shape


# In[5]:


# tSNE raw data in original (all) feature space
tsne = TSNE(n_components=2, init='pca', random_state=123, perplexity=30, learning_rate=300, n_iter=400)
tsne_X = tsne.fit_transform(X)
tsne_X = pd.DataFrame(tsne_X, columns=['1', '2'])
tsne_X.index = X.index
tsne_X.index.name = 'sample_id'       


# In[6]:


# Map sample id to clinial phenotype (i.e. experimental condition)

# Note:
# According to the source (https://github.com/scikit-learn/scikit-learn/blob/14031f6/sklearn/decomposition/pca.py#L310),
# input will be transformed by np.array() before doing PCA. So row index will be lost during 
# PCA.fit_transform(X) even using a structured array or a pandas DataFrame. However, the order of the data is preserved, 
# meaning you can attach the index back afterward

# read in mapping file (sample id --> phenotype)
mapper = pd.read_table(map_file, header=0, sep='\t', index_col=0)

# Join 
X_new = pd.merge(tsne_X, mapper, left_index=True, right_index=True)
X_new.head(5)
#X_new.shape


# In[7]:


# Plot
# Note: t-SNE has a cost function that is not convex, i.e. with different initializations we can get different results.

fg = sns.lmplot(x='1', y='2', data=X_new, hue='medium', fit_reg=False)
fg.add_legend()
fig_file = map_file = os.path.join(os.path.dirname(os.getcwd()), "viz", "{}.png".format(file_name))
fg.fig.suptitle(file_name.replace('_',' ').capitalize())
fg.savefig(fig_file)

# Plot for eADAGE doesn't have labels
#fg=sns.jointplot(x="1", y="2", data=tsne_X, kind='hex', stat_func=None);


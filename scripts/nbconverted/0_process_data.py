
# coding: utf-8

# # Process data
# 1. Examine the raw data
# 2. Determine normalization based on examination

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
from ggplot import *

from numpy.random import seed
randomState = 123
seed(randomState)


# In[2]:


# Load arguments
data_file = os.path.join(os.path.dirname(os.getcwd()), "data", "all-pseudomonas-gene.zip")
normalized_data_file = os.path.join(os.path.dirname(os.getcwd()), "data", "all-pseudomonas-gene-normalized.zip")
metadata_file = os.path.join(os.path.dirname(os.getcwd()), "metadata", "sample_annotations.tsv")


# In[3]:


# Read in data
data = pd.read_table(data_file, header=0, sep='\t', index_col=0, compression='zip').T
data.head(5)


# In[4]:


# Read in data
metadata = pd.read_table(metadata_file, header=0, sep='\t', index_col='ml_data_source')
metadata


# In[5]:


# Select metadata field
metadata_field = 'strain'
metadata_selected = metadata[metadata_field].to_frame()

metadata_selected.head(5)


# In[6]:


data_labeled = data.merge(metadata_selected, left_index=True, right_index=True, how='inner')
print(data_labeled.shape)
data_labeled.head(5)


# In[7]:


# UMAP embedding of raw gene space data
embedding = umap.UMAP().fit_transform(data_labeled.iloc[:,1:-1])
embedding_df = pd.DataFrame(data=embedding, columns=['1','2'])
embedding_df['metadata'] = list(data_labeled[metadata_field])
print(embedding_df.shape)
embedding_df


# In[8]:


# Plot
ggplot(aes(x='1',y='2', color='metadata'), data=embedding_df) + geom_point()
#plt.scatter(embedding[:, 0], embedding[:, 1]) #, c=[sns.color_palette()[x] for x in metadata.experiment])
#plt.gca().set_aspect('equal', 'datalim')
#plt.title('UMAP projection of the Iris dataset', fontsize=24);


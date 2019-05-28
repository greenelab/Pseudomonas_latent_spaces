
# coding: utf-8

# # Process data

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


# ## Data
# 
# Data downloaded from ADAGE repository [ADAGE](https://github.com/greenelab/adage).  
# 
# *data*
# 1. Raw data was downloaded from ArrayExpress
# 2. Use [RMA](https://www.rdocumentation.org/packages/affy/versions/1.50.0/topics/rma) bioconductor library to convert raw array data to log 2 gene expression data that has been quantile normalized.
# 3. Only keep PA genes, remove control genes
# 
# *normalized_data*
# 1. Use data from above
# 2. Normalize each gene to be between 0 and 1

# In[2]:


# Load arguments
data_file = os.path.join(os.path.dirname(os.getcwd()), "data", "Pa_compendium_02.22.2014.pcl")
normalized_data_file = os.path.join(os.path.dirname(os.getcwd()), "data", "train_set_normalized.pcl")
metadata_file = os.path.join(os.path.dirname(os.getcwd()), "metadata", "sample_annotations.tsv")


# In[3]:


# Read in data
data = pd.read_table(data_file, header=0, sep='\t', index_col=0).T
data.head(5)


# In[4]:


# Read in data
normalized_data = pd.read_table(normalized_data_file, header=0, sep='\t', index_col=0).T
normalized_data.head(5)


# In[5]:


# Read in metadata
metadata = pd.read_table(metadata_file, header=0, sep='\t', index_col='ml_data_source')
metadata


# ## Select metadata field

# In[6]:


# Select metadata field
# Choices:  medium
#           strain
#           growth_setting_1
#           experiment
metadata_field = 'growth_setting_1'
metadata_selected = metadata[metadata_field].to_frame()

metadata_selected.head(5)


# In[7]:


# Select subset of values within metadata field
if metadata_field == 'experiment':
    metadata_selected_labeled = metadata_selected

if metadata_field == 'strain':
    values = ['PAO1', 'PA14', 'CF sputum isolate']
    # Re-label metadata field based on subset of values
    metadata_vector = metadata[metadata_field]

    sample_id = metadata_selected.index

    metadata_selected_labeled = metadata_selected.assign(
        strain=(
            list( 
                map(
                    lambda x:
                    values[0] if x.lower().strip() == values[0].lower() 
                    else values[1] if x.lower().strip() == values[1].lower()
                    else values[2] if x.lower().strip() == values[2].lower()
                    else 'NA',
                    metadata_vector
                )      
            )
        )
    )
    metadata_selected_labeled = metadata_selected_labeled.astype({metadata_field: str})
    
if metadata_field == 'growth_setting_1':
    values = ['Planktonic', 'Biofilm', 'Colony', 'Swarm']
    
    # Re-label metadata field based on subset of values
    metadata_vector = metadata[metadata_field]

    sample_id = metadata_selected.index

    metadata_selected_labeled = metadata_selected.assign(
        growth_setting_1=(
            list( 
                map(
                    lambda x:
                    'NA' if type(x) == float
                    else values[0] if x.lower().strip() == values[0].lower() 
                    else values[1] if x.lower().strip() == values[1].lower()
                    else values[2] if x.lower().strip() == values[2].lower()
                    else values[3] if x.lower().strip() == values[3].lower()
                    else 'NA',
                    metadata_vector
                )      
            )
        )
    )
    metadata_selected_labeled = metadata_selected_labeled.astype({metadata_field: str})
    
if metadata_field == 'medium':
    values = ['LB', 'PIA', 'CF sputum', 'pyruvate', 'casamino acids']
    
    # Re-label metadata field based on subset of values
    metadata_vector = metadata[metadata_field]

    sample_id = metadata_selected.index

    metadata_selected_labeled = metadata_selected.assign(
        medium=(
            list( 
                map(
                    lambda x:
                    values[0] if values[0].lower() in x.lower().strip()
                    else values[1] if x.lower().strip() == values[1].lower()
                    else values[2] if values[2].lower() in x.lower().strip()
                    else values[3] if values[3].lower() in x.lower().strip()
                    else values[4] if values[4].lower() in x.lower().strip()
                    else 'NA',
                    metadata_vector
                )      
            )
        )
    )
    metadata_selected_labeled = metadata_selected_labeled.astype({metadata_field: str})
    


# In[8]:


metadata_selected_labeled


# In[9]:


metadata_selected


# ## Plot RMA normalized data

# In[10]:


data_labeled = data.merge(metadata_selected_labeled, left_index=True, right_index=True, how='inner')
print(data_labeled.shape)
data_labeled.head(5)


# In[11]:


# UMAP embedding of raw gene space data
embedding = umap.UMAP().fit_transform(data_labeled.iloc[:,1:-1])
embedding_df = pd.DataFrame(data=embedding, columns=['1','2'])
embedding_df['metadata'] = list(data_labeled[metadata_field])
print(embedding_df.shape)
embedding_df.head(5)


# In[12]:


# Plot
ggplot(aes(x='1',y='2', color='metadata'), data=embedding_df) +         geom_point(alpha=0.5) +         scale_color_brewer(type='qual', palette='Set2')
#ggMarginal(fig, type='histogram')


# ## Plot 0-1 normalized data

# In[13]:


normalized_data_labeled = normalized_data.merge(metadata_selected_labeled, left_index=True, right_index=True, how='inner')
print(normalized_data_labeled.shape)
normalized_data_labeled.head(5)


# In[14]:


# UMAP embedding of raw gene space data
embedding_normalized = umap.UMAP().fit_transform(normalized_data_labeled.iloc[:,1:-1])
embedding_normalized_df = pd.DataFrame(data=embedding_normalized, columns=['1','2'])
embedding_normalized_df['metadata'] = list(normalized_data_labeled[metadata_field])
print(embedding_normalized_df.shape)
embedding_normalized_df.head(5)


# In[15]:


# Plot
ggplot(aes(x='1',y='2', color='metadata'), data=embedding_df) +     geom_point(alpha=0.5) +     scale_color_brewer(type='qual', palette='Set2')


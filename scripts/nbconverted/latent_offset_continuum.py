
# coding: utf-8

# In[1]:


#-------------------------------------------------------------------------------------------------------------------------------
# By Alexandra Lee (July 2018) 
#
# Take the difference of the encoded gene expression of the two extreme experimental conditions
# This will be the offset for the latent space
#-------------------------------------------------------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
np.random.seed(123)


# In[2]:


# load arguments
max_file = os.path.join(os.path.dirname(os.getcwd()), "encoded", "oxygen_level", "train_maxO2_2layer_10latent_encoded.txt")
min_file = os.path.join(os.path.dirname(os.getcwd()), "encoded", "oxygen_level", "train_minO2_2layer_10latent_encoded.txt")

# output files
out_file = os.path.join(os.path.dirname(os.getcwd()), "data", "oxygen_level", "train_offset_2layer_10latent.txt")


# In[3]:


# read in data
max_data = pd.read_table(max_file, header = 0, sep = '\t', index_col = 0)
min_data = pd.read_table(min_file, header = 0, sep = '\t', index_col = 0)

min_data


# In[4]:


max_data


# In[5]:


# Generate offset using average gene expression in original dataset
train_offset_latent = min_data.values - max_data.values
train_offset_latent = pd.DataFrame(train_offset_latent, columns = min_data.columns)
train_offset_latent


# In[6]:


# output
train_offset_latent.to_csv(out_file, sep='\t')


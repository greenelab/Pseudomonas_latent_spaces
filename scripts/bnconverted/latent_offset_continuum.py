
# coding: utf-8

# In[2]:


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


# In[3]:


# load arguments
max_file = os.path.join(os.path.dirname(os.getcwd()), "encoded", "tybalt_2layer_10_train_anr_maxO2_encoded.txt")
min_file = os.path.join(os.path.dirname(os.getcwd()), "encoded", "tybalt_2layer_10_train_anr_minO2_encoded.txt")

# output files
out_file = os.path.join(os.path.dirname(os.getcwd()), "data", "train_offset_latent_2layer_anr.txt")


# In[4]:


# read in data
max_data = pd.read_table(max_file, header = 0, sep = '\t', index_col = 0)
min_data = pd.read_table(min_file, header = 0, sep = '\t', index_col = 0)

min_data


# In[5]:


max_data


# In[7]:


# Generate offset using average gene expression in original dataset
train_offset_latent = min_data.values - max_data.values
train_offset_latent = pd.DataFrame(train_offset_latent, columns = min_data.columns)
train_offset_latent


# In[6]:


# output
train_offset_latent.to_csv(out_file, sep='\t')


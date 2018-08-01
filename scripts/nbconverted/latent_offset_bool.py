
# coding: utf-8

# In[1]:


#-------------------------------------------------------------------------------------------------------------------------------
# By Alexandra Lee (July 2018) 
#
# Take the average of the encoded gene expression for the two experimental conditions
# Take the difference of the averages -- this will be the offset for the latent space
#-------------------------------------------------------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np

randomState = 123
from numpy.random import seed
seed(randomState)


# In[2]:


# load arguments
encodedA_file = os.path.join(os.path.dirname(os.getcwd()), "encoded", "train_treat_1layer_10latent_encoded.txt")
encodedB_file = os.path.join(os.path.dirname(os.getcwd()), "encoded", "train_control_1layer_10latent_encoded.txt")

# output files
out_file = os.path.join(os.path.dirname(os.getcwd()), "data", "train_offset_1layer_10latent.txt")


# In[3]:


# read in data
encodedA_data = pd.read_table(encodedA_file, header = 0, sep = '\t', index_col = 0)
encodedB_data = pd.read_table(encodedB_file, header = 0, sep = '\t', index_col = 0)
encodedA_data.head(5)


# In[4]:


# Average gene expression across samples in training set
train_A_mean = encodedA_data.mean(axis=0)
train_B_mean = encodedB_data.mean(axis=0)

train_A_mean


# In[5]:


train_B_mean


# In[6]:


# Generate offset using average gene expression in original dataset
train_offset_latent = train_A_mean - train_B_mean


train_offset_latent_df = pd.Series.to_frame(train_offset_latent).transpose()
train_offset_latent_df


# In[7]:


# output
train_offset_latent_df.to_csv(out_file, sep='\t')


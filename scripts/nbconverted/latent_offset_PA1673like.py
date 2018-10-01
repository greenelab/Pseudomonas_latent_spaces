
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
lowest_file = os.path.join(os.path.dirname(os.getcwd()), "encoded", "PA1673_full_old", "train_lowest_2layer_10latent_encoded.txt")
highest_file = os.path.join(os.path.dirname(os.getcwd()), "encoded", "PA1673_full_old", "train_highest_2layer_10latent_encoded.txt")

# output files
out_file = os.path.join(os.path.dirname(os.getcwd()), "data", "PA1673_full_old", "train_offset_2layer_10latent.txt")


# In[3]:


# read in data
lowest_data = pd.read_table(lowest_file, header = 0, sep = '\t', index_col = 0)
highest_data = pd.read_table(highest_file, header = 0, sep = '\t', index_col = 0)
lowest_data.head(5)


# In[4]:


highest_data.head(5)


# In[5]:


# Average gene expression across samples in training set
train_lowest_mean = lowest_data.mean(axis=0)
train_highest_mean = highest_data.mean(axis=0)

train_lowest_mean


# In[6]:


train_highest_mean


# In[7]:


# Generate offset using average gene expression in original dataset
train_offset_latent = train_highest_mean - train_lowest_mean


train_offset_latent_df = pd.Series.to_frame(train_offset_latent).transpose()
train_offset_latent_df


# In[8]:


# output
train_offset_latent_df.to_csv(out_file, sep='\t')


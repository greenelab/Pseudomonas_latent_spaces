
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
np.random.seed(123)


# In[2]:


# load arguments
encodedA_file = os.path.join(os.path.dirname(os.getcwd()), "encoded", "tybalt_2layer_10_train_treat_encoded.txt")
encodedB_file = os.path.join(os.path.dirname(os.getcwd()), "encoded", "tybalt_2layer_10_train_control_encoded.txt")

# output files
out_file = os.path.join(os.path.dirname(os.getcwd()), "data", "train_offset_latent_2layer.txt")


# In[3]:


# read in data
encodedA_data = pd.read_table(encodedA_file, header = 0, sep = '\t', index_col = 0)
encodedB_data = pd.read_table(encodedB_file, header = 0, sep = '\t', index_col = 0)
encodedA_data.head(5)


# In[4]:


# Change index names to integer for downstream sorting
encodedA_data.columns = [str(i) for i in list(range(0,10))]
encodedB_data.columns = [str(i) for i in list(range(0,10))]


# In[5]:


# Average gene expression across samples in training set
train_A_mean = encodedA_data.mean(axis=0)
train_B_mean = encodedB_data.mean(axis=0)

# Generate offset using average gene expression in original dataset
train_offset_latent = (train_A_mean - train_B_mean).sort_index(ascending=True)
train_offset_latent = pd.DataFrame(train_offset_latent, index = train_offset_latent.index)
train_offset_latent


# In[6]:


# output
train_offset_latent.to_csv(out_file, sep='\t')


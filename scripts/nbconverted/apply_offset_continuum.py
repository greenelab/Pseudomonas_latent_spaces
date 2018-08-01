
# coding: utf-8

# In[1]:


#-------------------------------------------------------------------------------------------------------------------------------
# By Alexandra Lee (July 2018) 
#
# Apply offset vector
#
# In original space: Add offset vector to each sample in the test set condition A to transform the gene expression 
# profile of the test samples to look like the samples are under condition B
#
# In latent space:  Add offset vector to each sample in the encoded test set condition A to transform the gene 
# expression profile of the test samples to look like the samples are under condition B
#-------------------------------------------------------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np

randomState = 123
from numpy.random import seed
seed(randomState)


# In[2]:


# load arguments
test_file = os.path.join(os.path.dirname(os.getcwd()), "encoded", "tybalt_2layer_10_test_anr_t5_encoded.txt")
offset_file = os.path.join(os.path.dirname(os.getcwd()), "data", "train_offset_latent_2layer_anr.txt")

# Are you applying the offset in the latent space?
latent = True

# Percentage of the offset to apply to the dataset
percentage = 0.95

# output files
out_file = os.path.join(os.path.dirname(os.getcwd()), "encoded", "estimated_test_t90_encoded.txt")


# In[3]:


# read in data
test_data = pd.read_table(test_file, header = 0, sep = '\t', index_col = 0)

# save header to attach back later
header = test_data.index

test_data


# In[4]:


# read offset
if latent:
    offset_data = pd.read_table(offset_file, header = 0, sep = '\t', index_col = 0)
    offset_data.index = [str(i) for i in offset_data.index]  # match index between test_data and offset_data
else:
    offset_data = pd.read_table(offset_file, header = 0, sep = '\t', index_col = 0)
    
#offset_data.index
offset_data


# In[5]:


# Apply offset
estimated_data = test_data.values + percentage*offset_data.values
estimated_data = pd.DataFrame(estimated_data, index = test_data.index)

estimated_data


# In[6]:


# Output estimated gene experession values
estimated_data.to_csv(out_file, sep='\t')


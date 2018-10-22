
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
test_file = os.path.join(os.path.dirname(os.getcwd()), "data", "cipro_treatment", "test_control.txt")
offset_file = os.path.join(os.path.dirname(os.getcwd()), "data", "cipro_treatment", "train_offset_original.txt")

# output files
out_file = os.path.join(os.path.dirname(os.getcwd()), "output", "cipro_treatment", "estimated_test_control_original.txt")


# In[3]:


# read in data
test_data = pd.read_table(test_file, header=0, sep='\t', index_col=0)

test_data


# In[4]:


# read offset
offset_data = pd.read_table(offset_file, header=0, sep='\t', index_col=0)

offset_data


# In[5]:


# Apply offset
estimated_data = test_data.values + offset_data.values
estimated_data = pd.DataFrame(estimated_data, index = test_data.index)

estimated_data


# In[6]:


# Output estimated gene experession values
estimated_data.to_csv(out_file, sep='\t')


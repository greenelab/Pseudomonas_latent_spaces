
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
np.random.seed(123)


# In[2]:


# load arguments
test_file = os.path.join(os.path.dirname(os.getcwd()), "data", "test_control.txt")
offset_file = os.path.join(os.path.dirname(os.getcwd()), "data", "train_offset_original.txt")

# Are you applying the offset in the latent space?
latent = False

# output files
out_file = os.path.join(os.path.dirname(os.getcwd()), "output", "estimated_test_control_original.txt")


# In[3]:


# read in data
test_data = pd.read_table(test_file, header = 0, sep = '\t', index_col = 0).transpose()

# save header to attach back later
header = test_data.columns

test_data.head(5)
#header


# In[4]:


# read offset
if latent:
    offset_data = pd.read_table(offset_file, header = 0, sep = '\t', index_col = 0)
    offset_data.index = [str(i) for i in offset_data.index]  # match index between test_data and offset_data
else:
    offset_data = pd.read_table(offset_file, header = None, sep = '\t', index_col = 0)
    
#offset_data.index
offset_data


# In[5]:


# Rename header to match
offset_data.columns = ['gene_exp']
test_data.columns = ['gene_exp']*test_data.shape[1]

test_data


# In[6]:


# Apply offset
estimated_data = test_data.add(offset_data, axis = 'index')
estimated_data.columns = header
estimated_data = estimated_data.transpose()

estimated_data


# In[7]:


# Output estimated gene experession values
estimated_data.to_csv(out_file, sep='\t')


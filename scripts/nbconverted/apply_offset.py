
# coding: utf-8

# In[1]:


#-------------------------------------------------------------------------------------------------------------------------------
# By Alexandra Lee (July 2018) 
#
# Apply offset vector
#-------------------------------------------------------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
np.random.seed(123)


# In[2]:


# load arguments
test_file = os.path.join(os.path.dirname(os.getcwd()), "encoded", "encoded_test_B.txt")
offset_file = os.path.join(os.path.dirname(os.getcwd()), "data", "train_offset_latent.txt")
latent = True

# output files
out_file = os.path.join(os.path.dirname(os.getcwd()), "estimated_geneExp", "estimated_test_B.txt")


# In[3]:


# read in data
test_data = pd.read_table(test_file, header = 0, sep = '\t', index_col = 0).transpose()

# save header to attach back later
header = test_data.columns

test_data.head(5)
test_data.index


# In[4]:


# read offset
offset_data = pd.read_table(offset_file, header = 0, sep = '\t', index_col = 0)
if latent:
    offset_data.index = [str(i) for i in offset_data.index]  # match index between test_data and offset_data
offset_data.head(5)


# In[5]:


# Rename header to match
offset_data.columns = ['gene_exp']
test_data.columns = ['gene_exp']*test_data.shape[1]


# In[6]:


# Apply offset
estimated_data = test_data.add(offset_data, axis = 'index')
estimated_data.columns = header
estimated_data


# In[7]:


# Output estimated gene experession values
estimated_data.to_csv(out_file, sep='\t')



# coding: utf-8

# In[14]:


#-------------------------------------------------------------------------------------------------------------------------------
# By Alexandra Lee (July 2018) 
#
# Calculate error between reconstructed vs observed gene expression
#-------------------------------------------------------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
np.random.seed(123)


# In[15]:


# load arguments
estimated_file = os.path.join(os.path.dirname(os.getcwd()), "estimated_geneExp", "decoded_test_B.txt")
obsv_file = os.path.join(os.path.dirname(os.getcwd()), "data", "test_A.txt")


# In[16]:


# read in data
estimated_data = pd.read_table(estimated_file, header = 0, sep = '\t', index_col = 0)
obsv_data = pd.read_table(obsv_file, header = 0, sep = '\t', index_col = 0).transpose()
obsv_data.head(5)


# In[17]:


# Calculate error: RMSE of estimated data and observed data per sample

# Note: estiamted and observed samples are not matched, so how do we compare them?
rmse = np.ndarray(shape=(1, estimated_data.shape[1]))
i = 0
for col in estimated_data.columns:
    rmse[0,i] = ((estimated_data[col] - obsv_data[col]) ** 2).mean() ** .5
    i+=1
rmse


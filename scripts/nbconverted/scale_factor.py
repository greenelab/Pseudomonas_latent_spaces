
# coding: utf-8

# In[1]:


# Determine scale factor for offset
# scale factor = avg PA1673 exp @ new location - avg PA1673 exp @ baseline  
import os
import pandas as pd
import numpy as np

randomState = 123
from numpy.random import seed
seed(randomState)


# In[2]:


# load arguments
baseline_file = os.path.join(os.path.dirname(os.getcwd()), "data", "PA1673_full_old", "test_lowest_PA1673.txt")
new_file = os.path.join(os.path.dirname(os.getcwd()), "data", "PA1673_full_old", "test_mid2_PA1673.txt")


# In[3]:


# read in data
baseline_data = pd.read_table(baseline_file, header = 0, sep = '\t', index_col = 0)
new_data = pd.read_table(new_file, header = 0, sep = '\t', index_col = 0)


# In[4]:


# average PA1673 expression across samples
baseline_mean = baseline_data['PA1673'].mean()
new_mean = new_data['PA1673'].mean()

baseline_mean


# In[5]:


new_mean


# In[6]:


# Get scale factor
scale_factor = new_mean - baseline_mean
print(scale_factor)


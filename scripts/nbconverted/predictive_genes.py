
# coding: utf-8

# In[1]:


#-------------------------------------------------------------------------------------------------------------------------------
# By Alexandra Lee (July 2018) 
#
# Identify genes that are predictive using latent transformation vs linear transformation
#-------------------------------------------------------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import seaborn as sns

randomState = 123
from numpy.random import seed
seed(randomState)


# In[2]:


# load arguments
estimated_latent_file = os.path.join(os.path.dirname(os.getcwd()), "output", "oxygen_level", "estimated_test_t90_2layer_10latent.txt")
estimated_original_file = os.path.join(os.path.dirname(os.getcwd()), "output", "oxygen_level", "estimated_test_t90_original.txt")
obsv_file = os.path.join(os.path.dirname(os.getcwd()), "data", "oxygen_level", "train_minO2.txt")


# In[3]:


# read in data
estimated_latent_data = pd.read_table(estimated_latent_file, header = 0, sep = '\t', index_col = 0)
estimated_original_data = pd.read_table(estimated_original_file, header = 0, sep = '\t', index_col = 0)
obsv_data = pd.read_table(obsv_file, header = 0, sep = '\t', index_col = 0)

estimated_latent_data


# In[4]:


estimated_original_data


# In[5]:


obsv_data


# In[6]:


# Latent space residuals 

# Format
estimated_latent_df = pd.DataFrame(estimated_latent_data.values.transpose(), index = obsv_data.columns, columns=['estimated'])
obsv_df = pd.DataFrame(obsv_data.values.transpose(), index = obsv_data.columns, columns = ['observed'])

# Join 
X = pd.merge(estimated_latent_df, obsv_df, left_index=True, right_index=True)

# Residuals: observed - estimated
X['residuals'] = X['observed']-X['estimated']
X.head(5)

# Mean and stdev of residuals
residual_mean = X['residuals'].values.mean()
residual_std = X['residuals'].values.std()

print("Mean of residuals is: {} \n Standard deviation of residuals is: {}".format(residual_mean, residual_std))

# Identify those genes that have a residual that are less than 1 std from mean
# Genes that are predictive using latent space
threshold = residual_std*3
latent_predictive_geneID = X.index[X['residuals']>= threshold].tolist()
#latent_predictive_geneID


# In[7]:


# Linear space residuals 

# Format
estimated_original_df = pd.DataFrame(estimated_original_data.values.transpose(), index = obsv_data.columns, columns=['estimated'])
obsv_df = pd.DataFrame(obsv_data.values.transpose(), index = obsv_data.columns, columns = ['observed'])

# Join 
Y = pd.merge(estimated_latent_df, obsv_df, left_index=True, right_index=True)

# Residuals: observed - estimated
Y['residuals'] = X['observed']-X['estimated']
Y.head(5)

# Mean and stdev of residuals
residual_mean = Y['residuals'].values.mean()
residual_std = Y['residuals'].values.std()

print("Mean of residuals is: {} \n Standard deviation of residuals is: {}".format(residual_mean, residual_std))

# Identify those genes that have a residual that are exceed 3 std from mean 
# Genes that are not predictive using linear space
threshold = residual_std*3
linear_notPredictive_geneID = X.index[X['residuals']>= threshold].tolist()
#linear_notPredictive_geneID


# In[8]:


# Intersection of lists
# Genes that are predictive in latent space and NOT predictive in linear space
target_geneID = list(set(latent_predictive_geneID) & set(linear_notPredictive_geneID))
print("Number of target genes is {}".format(len(target_geneID)))



# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#-------------------------------------------------------------------------------------------------------------------------------
# By Alexandra Lee (July 2018) 
#
# Plot correlation between reconstructed vs observed gene expression
#-------------------------------------------------------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import seaborn as sns

randomState = 123
from numpy.random import seed
seed(randomState)


# In[3]:


# load arguments
estimated_file = os.path.join(os.path.dirname(os.getcwd()), "output", "PA1673_full_old", "estimated_test_mid2_2layer_10latent.txt")
obsv_file = os.path.join(os.path.dirname(os.getcwd()), "data", "PA1673_full_old", "test_mid2_PA1673.txt")

# output
fig_file = os.path.join(os.path.dirname(os.getcwd()), "viz", "PA1673_full_old", "Latent_mid2.png")


# In[4]:


# read in data
estimated_data = pd.read_table(estimated_file, header = 0, sep = '\t', index_col = 0)
obsv_data = pd.read_table(obsv_file, header = 0, sep = '\t', index_col = 0)

estimated_data.columns = obsv_data.columns  #Add gene ids to merge on later

#obsv_data
estimated_data


# In[5]:


# Average gene expression across samples
estimated_data_mean = estimated_data.mean(axis=0)
obsv_data_mean = obsv_data.mean(axis=0)

estimated_data_mean_df = pd.DataFrame(estimated_data_mean, index = estimated_data_mean.index, columns=['estimated'])
obsv_data_mean_df = pd.DataFrame(obsv_data_mean, index = obsv_data_mean.index, columns = ['observed'])

estimated_data_mean_df.head(5)
#obsv_data_mean_df.head(5)


# In[6]:


# Join 
X = pd.merge(estimated_data_mean_df, obsv_data_mean_df, left_index=True, right_index=True)
X.head(5)


# In[7]:


# Plot
fg=sns.jointplot(x='estimated', y='observed', data=X, kind='hex');
fg.savefig(fig_file)


# In[8]:


# Calculate error: RMSE of estimated data and observed data per sample

# Note: estiamted and observed samples are not matched, so how do we compare them?
#rmse = np.ndarray(shape=(1, estimated_data.shape[1]))
#i = 0
#for col in estimated_data.columns:
#    rmse[0,i] = ((estimated_data[col] - obsv_data[col]) ** 2).mean() ** .5
#    i+=1
#rmse


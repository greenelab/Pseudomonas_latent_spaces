
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#-------------------------------------------------------------------------------------------------------------------------------
# By Alexandra Lee (July 2018) 
#
# Identify genes that are predictive using latent transformation vs linear transformation
# 
# To determine which genes are "predictive" we compare the gene expression predicted after performing a
# transformation in gene space with the actual gene expression.  If the correlation between the two measurements
# is high then this gene is considered "predictive" using the gene space model.  The same argument is made
# for using the latent space model
#-------------------------------------------------------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

randomState = 123
from numpy.random import seed
seed(randomState)


# In[3]:


# load arguments
estimated_latent_file = os.path.join(os.path.dirname(os.getcwd()), "output", "oxygen_level", "estimated_test_t90_2layer_10latent.txt")
estimated_original_file = os.path.join(os.path.dirname(os.getcwd()), "output", "oxygen_level", "estimated_test_t90_original.txt")
obsv_file = os.path.join(os.path.dirname(os.getcwd()), "data", "oxygen_level", "train_minO2.txt")

# output
fig_file = os.path.join(os.path.dirname(os.getcwd()), "viz", "oxygen_level", "predictive_genes.png")

both_geneID_file = os.path.join(os.path.dirname(os.getcwd()), "output", "oxygen_level", "both_geneID.txt")
none_geneID_file = os.path.join(os.path.dirname(os.getcwd()), "output", "oxygen_level", "none_geneID.txt")
latent_geneID_file = os.path.join(os.path.dirname(os.getcwd()), "output", "oxygen_level", "latent_geneID.txt")
original_geneID_file = os.path.join(os.path.dirname(os.getcwd()), "output", "oxygen_level", "original_geneID.txt")


# In[4]:


# read in data
estimated_latent_data = pd.read_table(estimated_latent_file, header = 0, sep = '\t', index_col = 0)
estimated_original_data = pd.read_table(estimated_original_file, header = 0, sep = '\t', index_col = 0)
obsv_data = pd.read_table(obsv_file, header = 0, sep = '\t', index_col = 0)

estimated_latent_data


# In[5]:


estimated_original_data


# In[6]:


obsv_data


# In[7]:


# Latent space residuals 

# Format
estimated_latent_df = pd.DataFrame(estimated_latent_data.values.transpose(), index = obsv_data.columns, columns=['estimated'])
obsv_df = pd.DataFrame(obsv_data.values.transpose(), index = obsv_data.columns, columns = ['observed'])

# Join 
X = pd.merge(estimated_latent_df, obsv_df, left_index=True, right_index=True)

# Residuals: observed - estimated
X['residuals_latent'] = abs(X['observed']-X['estimated'])
X.head(5)

# Mean and stdev of residuals
#residual_mean = X['residuals'].values.mean()
#residual_std = X['residuals'].values.std()

#print("Mean of residuals is: {} \n Standard deviation of residuals is: {}".format(residual_mean, residual_std))

# Identify those genes that have a residual that are less than 1 std from mean
# Genes that are predictive using latent space
#threshold = residual_std*3
#latent_predictive_geneID = X.index[X['residuals']>= threshold].tolist()
#latent_predictive_geneID


# In[8]:


# Linear space residuals 

# Format
estimated_original_df = pd.DataFrame(estimated_original_data.values.transpose(), index = obsv_data.columns, columns=['estimated'])
obsv_df = pd.DataFrame(obsv_data.values.transpose(), index = obsv_data.columns, columns = ['observed'])

# Join 
Y = pd.merge(estimated_latent_df, obsv_df, left_index=True, right_index=True)

# Residuals: observed - estimated
Y['residuals_original'] = abs(Y['observed']-Y['estimated'])
Y.head(5)

# Mean and stdev of residuals
#residual_mean = Y['residuals'].values.mean()
#residual_std = Y['residuals'].values.std()

#print("Mean of residuals is: {} \n Standard deviation of residuals is: {}".format(residual_mean, residual_std))

# Identify those genes that have a residual that are exceed 3 std from mean 
# Genes that are not predictive using linear space
#threshold = residual_std*3
#linear_notPredictive_geneID = X.index[X['residuals']>= threshold].tolist()
#linear_notPredictive_geneID


# In[9]:


# Join 
Z = pd.merge(X, Y, left_index=True, right_index=True)
Z.head(5)


# In[10]:


# Threshold
# Set threshold based on the dropoff after the shoulder from manual visual inspection
latent_threshold = 0.03
original_threshold = 0.03


# In[11]:


# Plot
fg=sns.jointplot(x='residuals_original', y='residuals_latent', data=Z, kind='hex');
fg.ax_joint.axhline(y=latent_threshold, color='k', linestyle='--')
fg.ax_joint.axvline(x=original_threshold, color='k', linestyle='--',)
fg.savefig(fig_file)


# In[12]:


# Intersection of lists
# Genes that are predictive in latent space and NOT predictive in linear space

latent_predictive = Z.index[Z['residuals_latent'] < latent_threshold].tolist()
latent_notPredictive = Z.index[Z['residuals_latent'] > latent_threshold].tolist()
original_predictive = Z.index[Z['residuals_original'] < original_threshold].tolist()
original_notPredictive = Z.index[Z['residuals_original'] > original_threshold].tolist()

both_geneID = list(set(latent_predictive) & set(original_predictive))
none_geneID = list(set(latent_notPredictive) & set(original_notPredictive))
latent_geneID = list(set(latent_predictive) & set(original_notPredictive))
original_geneID = list(set(latent_notPredictive) & set(original_predictive))


print("Number of genes that are predictive in both spaces is {}".format(len(both_geneID)))
print("Number of genes that are predictive in neither space is {}".format(len(none_geneID)))
print("Number of genes that are predictive in latent space is {}".format(len(latent_geneID)))
print("Number of genes that are predictive in original space is {}".format(len(original_geneID)))

#linear_notPredictive_geneID = X.index[X['residuals']>= threshold].tolist()
#target_geneID = list(set(latent_predictive_geneID) & set(linear_notPredictive_geneID))
#print("Number of target genes is {}".format(len(target_geneID)))

# output list of genes to file
if len(both_geneID) > 0:
    pd.DataFrame(both_geneID).to_csv(both_geneID_file, sep='\t')
if len(none_geneID) > 0:
    pd.DataFrame(none_geneID).to_csv(none_geneID_file, sep='\t')
if len(latent_geneID) > 0:
    pd.DataFrame(latent_geneID).to_csv(latent_geneID_file, sep='\t')
if len(original_geneID) > 0:
    pd.DataFrame(original_geneID).to_csv(original_geneID_file, sep='\t')


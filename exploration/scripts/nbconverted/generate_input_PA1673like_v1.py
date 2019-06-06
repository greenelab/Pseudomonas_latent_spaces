
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#-------------------------------------------------------------------------------------------------------------------------------
# By Alexandra Lee (Septermber 2018) 
#
# Generate input files
#
# Dataset: Pseudomonas aeruginosa gene expression compendium referenced in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5069748/
# 
# Group samples based on gene expression of PA1673
#
# Generate offset vector using extreme gene expression values (train_offset_original):
# average highest gene expression - average lowest gene expression 
#-------------------------------------------------------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
from scipy.stats import variation
import seaborn as sns
import matplotlib.pyplot as plt

randomState = 123
from numpy.random import seed
seed(randomState)


# In[3]:


# Load arguments
data_file = os.path.join(os.path.dirname(os.getcwd()), "data", "all-pseudomonas-gene-normalized.zip")  # repo file is zipped
gene_id = "PA1673"

# Output training
train_highest_file = os.path.join(os.path.dirname(os.getcwd()), "data", "PA1673_full_old", "train_highest_PA1673.txt")
train_lowest_file = os.path.join(os.path.dirname(os.getcwd()), "data", "PA1673_full_old", "train_lowest_PA1673.txt")
train_input_file = os.path.join(os.path.dirname(os.getcwd()), "data", "PA1673_full_old", "train_model_input.txt.xz")

# Output test
test_lowest_file = os.path.join(os.path.dirname(os.getcwd()), "data", "PA1673_full_old", "test_lowest_PA1673.txt")
test_mid1_file = os.path.join(os.path.dirname(os.getcwd()), "data", "PA1673_full_old", "test_mid1_PA1673.txt")
test_mid2_file = os.path.join(os.path.dirname(os.getcwd()), "data", "PA1673_full_old", "test_mid2_PA1673.txt")

# Output offset
original_offset_file = os.path.join(os.path.dirname(os.getcwd()), "data", "PA1673_full_old", "train_offset_original.txt")


# In[4]:


# Read in data
data = pd.read_table(data_file, header=0, sep='\t', index_col=0, compression='zip')
X = data.transpose()
X.shape


# In[5]:


# Plot distribution of gene_id gene expression 
sns.distplot(X[gene_id])


# In[6]:


# Collect the extreme gene expressions
highest = X[X[gene_id] >= np.percentile(X[gene_id], 95)]
lowest = X[X[gene_id] <= np.percentile(X[gene_id], 5)]


# In[7]:


# Checkpoint
print(highest.shape)
print(lowest.shape)


# In[8]:


# Checkpoint
print(np.percentile(X[gene_id], 95))
print(highest[gene_id])
print(np.percentile(X[gene_id], 5))
print(lowest[gene_id])


# In[9]:


# Create dataframes with intermediate gene expression levels
baseline = X[(X[gene_id] > np.percentile(X[gene_id], 5)) & (X[gene_id] <= np.percentile(X[gene_id], 10))]
mid_1 = X[(X[gene_id] > np.percentile(X[gene_id], 40)) & (X[gene_id] <= np.percentile(X[gene_id], 50))]
mid_2 = X[(X[gene_id] > np.percentile(X[gene_id], 70)) & (X[gene_id] <= np.percentile(X[gene_id], 80))]


# In[10]:


# Checkpoint
print(baseline.shape)
print(mid_1.shape)
print(mid_2.shape)


# In[11]:


# Create input to VAE using all samples and holding out intermediate and baseline samples
input_holdout = (
    X
    .drop(baseline.index)
    .drop(mid_1.index)
    .drop(mid_2.index)
)


# In[12]:


# Checkpoint
print(X.shape)
print(input_holdout.shape)


# In[13]:


# Define offset vector using all genes
# Average gene expression across samples in training set
train_highest_mean = highest.mean(axis=0)
train_lowest_mean = lowest.mean(axis=0)

# Generate offset using average gene expression in original dataset
train_offset_original = train_highest_mean - train_lowest_mean
train_offset_original_df = pd.DataFrame(train_offset_original).transpose()
train_offset_original_df


# In[14]:


# Output training and test sets

# training data
highest.to_csv(train_highest_file, sep='\t')
lowest.to_csv(train_lowest_file, sep='\t')
input_holdout.to_csv(train_input_file, sep='\t', compression='xz')

# test data
baseline.to_csv(test_lowest_file, sep='\t')
mid_1.to_csv(test_mid1_file, sep='\t')
mid_2.to_csv(test_mid2_file, sep='\t')

# original offset
train_offset_original_df.to_csv(original_offset_file, sep='\t')


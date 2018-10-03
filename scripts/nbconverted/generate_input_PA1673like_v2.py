
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


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


# In[5]:


# Load arguments
data_file = os.path.join(os.path.dirname(os.getcwd()), "data", "all-pseudomonas-gene-normalized.zip")  # repo file is zipped
gene_id = "PA1673"

# Output training
train_highest_file = os.path.join(os.path.dirname(os.getcwd()), "data", "PA1673_full_test", "train_highest_PA1673.txt")
train_lowest_file = os.path.join(os.path.dirname(os.getcwd()), "data", "PA1673_full_test", "train_lowest_PA1673.txt")
train_mid1_file = os.path.join(os.path.dirname(os.getcwd()), "data", "PA1673_full_test", "train_mid1_PA1673.txt")
train_mid2_file = os.path.join(os.path.dirname(os.getcwd()), "data", "PA1673_full_test", "train_mid2_PA1673.txt")
train_input_file = os.path.join(os.path.dirname(os.getcwd()), "data", "PA1673_full_test", "train_model_input.txt.xz")

# Output test
test_lowest_file = os.path.join(os.path.dirname(os.getcwd()), "data", "PA1673_full_test", "test_lowest_PA1673.txt")
test_mid1_file = os.path.join(os.path.dirname(os.getcwd()), "data", "PA1673_full_test", "test_mid1_PA1673.txt")
test_mid2_file = os.path.join(os.path.dirname(os.getcwd()), "data", "PA1673_full_test", "test_mid2_PA1673.txt")
test_highest_file = os.path.join(os.path.dirname(os.getcwd()), "data", "PA1673_full_test", "test_highest_PA1673.txt")

# Output offset
original_offset_file = os.path.join(os.path.dirname(os.getcwd()), "data", "PA1673_full_test", "train_offset_original.txt")


# In[6]:


# Read in data
data = pd.read_table(data_file, header=0, sep='\t', index_col=0, compression='zip')
X = data.transpose()
X.shape


# In[7]:


# Plot distribution of gene_id gene expression 
sns.distplot(X[gene_id])


# In[8]:


# Collect the extreme gene expressions
highest = X[X[gene_id] >= np.percentile(X[gene_id], 95)]
lowest = X[X[gene_id] <= np.percentile(X[gene_id], 5)]


# In[9]:


# Checkpoint
print(highest.shape)
print(lowest.shape)


# In[17]:


# Checkpoint
#print(np.percentile(X[gene_id], 95))
#print(highest[gene_id])
#print(np.percentile(X[gene_id], 5))
print(lowest[gene_id])


# In[11]:


# Create dataframes with intermediate gene expression levels
mid_1 = X[(X[gene_id] > np.percentile(X[gene_id], 40)) & (X[gene_id] <= np.percentile(X[gene_id], 50))]
mid_2 = X[(X[gene_id] > np.percentile(X[gene_id], 70)) & (X[gene_id] <= np.percentile(X[gene_id], 80))]


# In[14]:


mid_1[gene_id]
#print(np.percentile(X[gene_id], 40))
#print(np.percentile(X[gene_id], 50))


# In[10]:


# Checkpoint
print(mid_1.shape)
print(mid_2.shape)


# In[11]:


# Partition the lowest, mid_1, mid_2, highest into training and test sets
# Training sets will be used to:
# (1) Train the VAE
# (2) Define the offset vectors
# (3) Define the scale factors for the offset

# Test sets will be used in the perturbation analysis

# Split 20% test set randomly
test_set_percent = 0.2
test_lowest = lowest.sample(frac=test_set_percent, random_state=randomState)
test_highest = highest.sample(frac=test_set_percent, random_state=randomState)
test_mid1 = mid_1.sample(frac=test_set_percent, random_state=randomState)
test_mid2 = mid_2.sample(frac=test_set_percent, random_state=randomState)

# Training sets
train_lowest = lowest.drop(test_lowest.index)
train_mid1 = mid_1.drop(test_mid1.index)
train_mid2 = mid_2.drop(test_mid2.index)
train_highest = highest.drop(test_highest.index)


# In[12]:


# Create input to VAE using all samples and holding out test sets
input_holdout = (
    X
    .drop(test_lowest.index)
    .drop(test_mid1.index)
    .drop(test_mid2.index)
    .drop(test_highest.index)
)


# In[13]:


# Checkpoint
print(X.shape)
print(input_holdout.shape)
print(test_lowest.shape)
print(test_mid1.shape)
print(test_mid2.shape)
print(test_highest.shape)


# In[14]:


# Define offset vector using all genes
# Average gene expression across samples in training set
train_highest_mean = train_highest.mean(axis=0)
train_lowest_mean = train_lowest.mean(axis=0)

# Generate offset using average gene expression in original dataset
train_offset_original = train_highest_mean - train_lowest_mean
train_offset_original_df = pd.DataFrame(train_offset_original).transpose()
train_offset_original_df


# In[15]:


# Output training and test sets

# training data
train_highest.to_csv(train_highest_file, sep='\t')
train_mid1.to_csv(train_mid1_file, sep='\t')
train_mid2.to_csv(train_mid2_file, sep='\t')
train_lowest.to_csv(train_lowest_file, sep='\t')
input_holdout.to_csv(train_input_file, sep='\t', compression='xz')

# test data
test_lowest.to_csv(test_lowest_file, sep='\t')
test_mid1.to_csv(test_mid1_file, sep='\t')
test_mid2.to_csv(test_mid2_file, sep='\t')
test_highest.to_csv(test_highest_file, sep='\t')

# original offset
train_offset_original.to_csv(original_offset_file, sep='\t')


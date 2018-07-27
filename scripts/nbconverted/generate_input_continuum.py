
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#-------------------------------------------------------------------------------------------------------------------------------
# By Alexandra Lee (July 2018) 
#
# Generate input files
#
# Dataset: Pseudomonas aeruginosa gene expression compendium referenced in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5069748/
# 
# Use map_file to group samples into phenotype groups (condition A and B) based on experimental design annotations
# Example: control vs treatment with antibiotics
# 
# Then group samples into training and test sets
#
# Generate offset vector using gene expression data in the original space (train_offset_original):
# average gene expression for condition A - average gene expression for condition B using all genes/dimensions
#-------------------------------------------------------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
from scipy.stats import variation
import seaborn as sns
import matplotlib.pyplot as plt
np.random.seed(123)


# In[3]:


# load arguments
#data_file = '/home/alexandra/Documents/Pseudomonas_scratch/all-pseudomonas-gene-normalized.pcl'
data_file = os.path.join(os.path.dirname(os.getcwd()), "data", "all-pseudomonas-gene-normalized.zip")  # repo file is zipped
map_file = os.path.join(os.path.dirname(os.getcwd()), "metadata", "mapping_anr.txt")


# In[4]:


# read in data
data = pd.read_table(data_file, header = 0, sep = '\t', index_col = 0, compression='zip')
X = data.transpose()
X.head(5)


# In[5]:


# read in metadata file containing grouping of each sample into training/test and phenotypic group
grp = pd.read_table(map_file, header = 0, sep = '\t', index_col = None)
grp


# In[6]:


# Group samples into training and test sets
# Training: min and max levels of O2
# Test: all intermediate levels

maxO2 = pd.DataFrame()
minO2 = pd.DataFrame()
intermediate = pd.DataFrame()

for index, row in grp.iterrows():
    if row['Group'] == 'Train':
        if row['Phenotype'] == "maxO2":
            sample = str(row['Sample ID'])
            maxO2 = maxO2.append(X[X.index.str.contains(sample, regex=False)])
        else:
            sample = str(row['Sample ID'])
            minO2 = minO2.append(X[X.index.str.contains(sample, regex=False)])
    if row['Group'] == 'Test':
        sample = str(row['Sample ID'])
        intermediate = intermediate.append(X[X.index.str.contains(sample, regex=False)])

#maxO2
#minO2
intermediate


# In[7]:


# Create input holding out test test (intermediate time points)
input_holdout = X.drop(intermediate.index)

input_holdout.shape
#X.shape


# In[8]:


# Generate offset using average gene expression in original dataset
train_offset_original = minO2.values - maxO2.values
train_offset_original = pd.DataFrame(train_offset_original, columns = minO2.columns)
train_offset_original


# In[9]:


# Output training and test sets

# training data
maxO2.to_csv(os.path.join(os.path.dirname(os.getcwd()), "data", "train_maxO2.txt"), sep='\t')
minO2.to_csv(os.path.join(os.path.dirname(os.getcwd()), "data", "train_minO2.txt"), sep='\t')
#input_holdout.to_csv(os.path.join(os.path.dirname(os.getcwd()), "data", "train_model_input_anr.txt.xz"), sep='\t', compression='xz')

# test data
for index, row in grp.iterrows():
    if row['Group'] == 'Test':
        sample = str(row['Sample ID'])
        df = pd.DataFrame(intermediate.loc[sample]).transpose()
        df.to_csv(os.path.join(os.path.dirname(os.getcwd()), "data", "test_"+row['Phenotype']+".txt"), sep='\t')

# original offset
train_offset_original.to_csv(os.path.join(os.path.dirname(os.getcwd()), "data", "train_offset_original_anr.txt"), sep='\t')


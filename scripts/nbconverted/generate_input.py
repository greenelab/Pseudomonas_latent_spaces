
# coding: utf-8

# In[1]:


#-------------------------------------------------------------------------------------------------------------------------------
# By Alexandra Lee (July 2018) 
#
# Generate input datasets
# Use map_file to group samples into phenotype groups (condition A and B) based on experimental design annotations
# Then group samples into training and test sets
#
# Generate offset vector using gene expression data in the original space (train_offset_original):
# average gene expression for condition A - average gene expression for condition B using all genes/dimensions
#-------------------------------------------------------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
np.random.seed(123)


# In[2]:


# load arguments
data_file = 'C:/Users/alexj/Documents/UPenn/CGreene/Pseudomonas_scratch/data/all-pseudomonas-gene-normalized.pcl'
#data_file = os.path.join(os.path.dirname(os.getcwd()), "data", "all-pseudomonas-gene-normalized.pcl")  # repo file is zipped
map_file = os.path.join(os.path.dirname(os.getcwd()), "metadata", "mapping_phosphate.txt")


# In[3]:


# read in data
data = pd.read_table(data_file, header = 0, sep = '\t', index_col = 0)
X = data.transpose()
X.head(5)


# In[4]:


# read in metadata file containing grouping of each sample into training/test and phenotypic group
grp = pd.read_table(map_file, header = 0, sep = '\t', index_col = None)
grp.head(5)


# In[5]:


# Group samples into condition A and B based on mapping file provided

# ***Group samples into training and test sets based on percentage***
train_A = pd.DataFrame()
train_B = pd.DataFrame()
test_A = pd.DataFrame()
test_B = pd.DataFrame()

for index, row in grp.iterrows():
    if row['Dataset'] == 'Train':
        if row['Group'] == 'A':
            sample = str(row['Sample ID'])
            train_A = train_A.append(X[X.index.str.match(sample)])
            #print('Training group A {}'.format(sample))
        else:
            sample = str(row['Sample ID'])
            train_B = train_B.append(X[X.index.str.match(sample)])
            #print('Training group B {}'.format(sample))
    else:
        if row['Group'] == 'A':
            sample = str(row['Sample ID'])
            test_A = test_A.append(X[X.index.str.match(sample)])
            #print('Test group A {}'.format(sample))
        else:
            sample = str(row['Sample ID'])
            test_B = test_B.append(X[X.index.str.match(sample)])
            #print('Test group B {}'.format(sample))
train_A


# In[6]:


# Average gene expression across samples in training set
train_A_mean = train_A.mean(axis=0)
train_B_mean = train_B.mean(axis=0)

# Generate offset using average gene expression in original dataset
train_offset_original = train_A_mean - train_B_mean


# In[7]:


# Output training and test sets
train_A.to_csv(os.path.join(os.path.dirname(os.getcwd()), "data", "train_A.txt"), sep='\t')
train_B.to_csv(os.path.join(os.path.dirname(os.getcwd()), "data", "train_B.txt"), sep='\t')
test_A.to_csv(os.path.join(os.path.dirname(os.getcwd()), "data", "test_A.txt"), sep='\t')
test_B.to_csv(os.path.join(os.path.dirname(os.getcwd()), "data", "test_B.txt"), sep='\t')

train_offset_original.to_csv(os.path.join(os.path.dirname(os.getcwd()), "data", "train_offset_original.txt"), sep='\t')


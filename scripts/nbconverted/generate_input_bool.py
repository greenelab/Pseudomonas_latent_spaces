
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
map_file = os.path.join(os.path.dirname(os.getcwd()), "metadata", "mapping_cipro.txt")


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


# Group samples into condition A and B based on mapping file provided
control_all = pd.DataFrame()
treat_all = pd.DataFrame()

for index, row in grp.iterrows():
    if row['Group'] == 'control':
        sample = str(row['Sample ID'])
        control_all = control_all.append(X[X.index.str.contains(sample, regex=False)])
        #print('Training group A {}'.format(sample))
    else:
        sample = str(row['Sample ID'])
        treat_all = treat_all.append(X[X.index.str.contains(sample, regex=False)])
        #print('Training group B {}'.format(sample))

# Split 10% test set randomly
test_set_percent = 0.2
test_control = control_all.sample(frac=test_set_percent)
train_control = control_all.drop(test_control.index)

test_treat = treat_all.sample(frac=test_set_percent)
train_treat = treat_all.drop(test_treat.index)

#control_all
#train_treat
#test_treat


# In[13]:


# Calculate Coefficient of Variance (CV) to determine variance between samples
# CV is the standardized measure of dispersion from the mean
treat_cv = variation(treat_all, axis = 0)
treat_cv = pd.DataFrame(treat_cv, columns = ['cv_treat'])
treat_cv.insert(0, 'gene_id', treat_all.columns)

control_cv = variation(control_all, axis = 0)
control_cv = pd.DataFrame(control_cv, columns = ['cv_control'])
control_cv.insert(0, 'gene_id', control_all.columns)

# Join 
CV = pd.merge(treat_cv, control_cv, on = 'gene_id')
CV = pd.melt(CV, id_vars = 'gene_id', var_name = 'group', value_name = 'cv' )
CV.head(5)

# figure
fig = plt.figure()
fg = sns.boxplot(x = 'group', y='cv', hue='group',  data=CV, palette="Set3")


# In[8]:


# Create input holding out test test
input_holdout = X.drop(test_control.index)
input_holdout = input_holdout.drop(test_treat.index)

input_holdout.head(5)
input_holdout.shape
#X.shape


# In[9]:


# Average gene expression across samples in training set
train_control_mean = train_control.mean(axis=0)
train_treat_mean = train_treat.mean(axis=0)

# Generate offset using average gene expression in original dataset
train_offset_original = train_treat_mean - train_control_mean


# In[10]:


# Output training and test sets
fig_file = os.path.join(os.path.dirname(os.getcwd()), "viz", "cv.png")
fig.savefig(fig_file)

#train_control.to_csv(os.path.join(os.path.dirname(os.getcwd()), "data", "train_control.txt"), sep='\t')
#train_treat.to_csv(os.path.join(os.path.dirname(os.getcwd()), "data", "train_treat.txt"), sep='\t')

#test_control.to_csv(os.path.join(os.path.dirname(os.getcwd()), "data", "test_control.txt"), sep='\t')
#test_treat.to_csv(os.path.join(os.path.dirname(os.getcwd()), "data", "test_treat.txt"), sep='\t')

#train_offset_original.to_csv(os.path.join(os.path.dirname(os.getcwd()), "data", "train_offset_original.txt"), sep='\t')
#input_holdout.to_csv(os.path.join(os.path.dirname(os.getcwd()), "data", "train_model_input.txt.xz"), sep='\t', compression='xz')

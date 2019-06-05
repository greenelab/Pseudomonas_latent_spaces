
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import random
import pickle
from sklearn.decomposition import PCA
from functions import utils

base_dirs = [os.path.join(os.path.dirname(os.getcwd()), 'data'),
             os.path.join(os.path.dirname(os.getcwd()), 'encoded'),
             os.path.join(os.path.dirname(os.getcwd()), 'models'),
             os.path.join(os.path.dirname(os.getcwd()), 'output'),
             os.path.join(os.path.dirname(os.getcwd()), 'stats'),
             os.path.join(os.path.dirname(os.getcwd()), 'viz') 
            ]
             
analysis_name = 'sim_1_test'
             
model_dir = os.path.join(base_dirs[2], analysis_name)
             
model_file = os.path.join(model_dir, "pca_model.pkl")

data_dir = os.path.join(base_dirs[0], analysis_name)

gene_id = 'PA0996'

percent_low = 5
percent_high = 95


# In[2]:


# Load arguments
target_gene_file = os.path.join(data_dir, gene_id + ".txt")
non_target_gene_file = os.path.join(data_dir, "train_model_input.txt.xz")

# Read in data
target_gene_data = pd.read_table(target_gene_file, header=0, index_col=0)
non_target_gene_data = pd.read_table(non_target_gene_file, header=0, index_col=0)
    
# Sort target gene data by expression (lowest --> highest)
target_gene_sorted = target_gene_data.sort_values(by=[gene_id])

# Collect the extreme gene expressions
[low_ids, high_ids] = utils.get_gene_expression_above_percent(target_gene_sorted, gene_id, percent_low, percent_high)
low_exp = non_target_gene_data.loc[low_ids]    
high_exp = non_target_gene_data.loc[high_ids]

print('Number of genes in low expression group is {}'.format(low_exp.shape))
print('Number of gene in high expression group is {}'.format(high_exp.shape))
    
# Load pca model
infile = open(model_file,'rb')
pca = pickle.load(infile)
infile.close()
    
# Transform data using loaded model
low_exp_encoded = pca.transform(low_exp)
high_exp_encoded = pca.transform(high_exp)

low_exp_encoded_df = pd.DataFrame(low_exp_encoded, index=low_exp.index)
high_exp_encoded_df = pd.DataFrame(high_exp_encoded, index=high_exp.index)


# In[3]:


low_exp_encoded_df


# In[4]:


high_exp_encoded_df


# In[6]:


# Average the gene expression transformed
lowest_mean = low_exp_encoded_df.mean(axis=0)
highest_mean = high_exp_encoded_df.mean(axis=0)

# Generate offset using average gene expression in original dataset
offset_latent_space = highest_mean - lowest_mean
offset_latent_space_df = pd.Series.to_frame(offset_latent_space).T
    
offset_latent_space_df


# In[7]:


# output lowest and highest expressing samples
low_exp_encoded_df.to_csv(lowest_file, sep='\t', float_format="%.5g")
high_exp_encoded_df.to_csv(highest_file, sep='\t', float_format="%.5g")

# ouput gene space offset vector
offset_latent_space_df.to_csv(offset_file, sep='\t', float_format="%.5g")


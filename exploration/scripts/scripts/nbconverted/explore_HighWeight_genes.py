
# coding: utf-8

# # Explore latent space features
# We want to know if our VAE model is capturing our signal
# 
# Identify high weight (HW) genes for each latent space feature (node)
# Determine if genes in group A and B are found within these highly weighted genes

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


randomState = 123
from numpy.random import seed
seed(randomState)


# In[2]:


# Load arguments
analysis_name = "sim_balancedAB_2latent"
data_file = os.path.join(os.path.dirname(os.getcwd()), "data", analysis_name, "VAE_weight_matrix.txt")
A_file = os.path.join(os.path.dirname(os.getcwd()), "data", analysis_name, "geneSetA.txt")
B_file = os.path.join(os.path.dirname(os.getcwd()), "data", analysis_name, "geneSetB.txt")

#HW_file = os.path.join(os.path.dirname(os.getcwd()), "output", analysis_name, "HW_genes.txt")


# In[3]:


# Read in data
geneSetA = pd.read_table(A_file, header=0, index_col=0)
geneSetB = pd.read_table(B_file, header=0, index_col=0)

weight = pd.read_table(data_file, header=0, index_col=0).T
weight.head(5)


# In[4]:


# Plot the distribution of gene weights per latent feature
#num_features = weight.shape[1]

#f, axes = plt.subplots(num_features, 1, sharex=True)
#for i, ax in zip(weight.columns, axes.flat):
#    sns.distplot(weight[i] , color="skyblue", rug=False, kde=False, ax=ax)
    
#f.set_size_inches(15, 15)


# In[5]:


# Calculate mean per node ("signature" or "feature")
means = weight.mean(axis=0)

# Calculate 2 standard deviations per node ("signature" or "feature")
stds = weight.std(axis=0)
two_stds = 2*stds


# In[6]:


# Get high positive and negative weight genes per node ("signature" or "feature") -- HW_df
# Record metadata per latent feature in a table -- HW_metadata_df
# Record distance between gene set weight and 
HW_df = pd.DataFrame()
HW_metadata_df = pd.DataFrame()

num_nodes = len(means)

for i in range(num_nodes):
    node_mean = means.iloc[i]
    node_std = stds.iloc[i] 
    
    pos_threshold = node_mean + two_stds.iloc[i]
    neg_threshold = node_mean - two_stds.iloc[i]
    
    hw_pos_genes = weight[weight[i] > pos_threshold].index
    hw_neg_genes = weight[weight[i] < neg_threshold].index
    
    node = str(i)
    node_name = 'Sig'+node+'pos'
    
    # Add high weight positive genes
    add_pos = pd.DataFrame({node_name: list(hw_pos_genes)})
    HW_df = pd.concat([HW_df, add_pos], axis=1)
    
    # Add metadata for positive node
    add_metadata_pos = pd.DataFrame({node_name: [node_mean, node_std, pos_threshold]})
    HW_metadata_df = pd.concat([HW_metadata_df, add_metadata_pos], axis=1)
    
    # Add high weight negative genes
    node_name = 'Sig'+node+'neg'
    
    add_neg = pd.DataFrame({node_name: list(hw_neg_genes)})
    HW_df = pd.concat([HW_df, add_neg], axis=1)
    
    # Add metadata for negative node
    add_metadata_neg = pd.DataFrame({node_name: [node_mean, node_std, neg_threshold]})
    HW_metadata_df = pd.concat([HW_metadata_df, add_metadata_neg], axis=1)

HW_df = HW_df.T
HW_metadata_df = HW_metadata_df.T


# In[7]:


HW_df.head()


# In[8]:


# Dataframe with the mean, std, threshold used per feature to determine high weight gene sets
HW_metadata_df.columns = ["mean", "std", "threshold"]
HW_metadata_df.head()


# In[9]:


# Create a table (feature x gene set A)
# Each cell will contain the distance between the weight and the mean

Weights_pos_neg_df = pd.DataFrame()

for i in range(num_nodes):
    node_mean = means.iloc[i]
    
    node = str(i+1)
    node_name = 'Sig'+node+'pos'
    
    gene_weight = weight[i]
    
    # Add high weight positive genes
    add_pos = pd.DataFrame({node_name: gene_weight})
    Weights_pos_neg_df = pd.concat([Weights_pos_neg_df, add_pos], axis=1)
    
    # Add high weight negative genes
    node_name = 'Sig'+node+'neg'
    
    add_neg = pd.DataFrame({node_name: gene_weight})
    Weights_pos_neg_df = pd.concat([Weights_pos_neg_df, add_neg], axis=1)
    
    
Weights_pos_neg_df = Weights_pos_neg_df.T
Weights_pos_neg_df.head()


# In[10]:


# Compare the weights for geneset A with threshold
geneSetA_ls = [l[0] for l in geneSetA.values.tolist()]
Weight_A = Weights_pos_neg_df[geneSetA_ls]

Weight_A.head()


# In[11]:


sns.distplot(Weight_A.iloc[0])


# In[12]:


# Compare the weights for geneset B with threshold
geneSetB_ls = [l[0] for l in geneSetB.values.tolist()]
Weight_B = Weights_pos_neg_df[geneSetB_ls]

Weight_B.head()


# In[13]:


sns.distplot(Weight_B.iloc[0])


# In[14]:


# What is the overlap between the high weight genes and gene sets A and B?
num_A = geneSetA.shape[0]
num_B = geneSetB.shape[0]

num_features_pos_neg = HW_df.shape[0]

percent_overlap = pd.DataFrame({'feature': [], 
                                'percent in A': [],
                                'percent in B': []
                               })

for i in range(num_features_pos_neg):
#for i in range(2):
    row = HW_df.iloc[i]
    percent_in_A = (row.isin(geneSetA_ls).sum()/num_A)*100
    percent_in_B = (row.isin(geneSetB_ls).sum()/num_B)*100
    
    add = pd.DataFrame({'feature': [HW_df.index[i]],
                        'percent in A': [percent_in_A],
                        'percent in B': [percent_in_B]
                       })
    percent_overlap = percent_overlap.append(add)

percent_overlap


# In[15]:


# Are there any features that are nonzero?
overlap_in_A = percent_overlap.iloc[percent_overlap["percent in A"].nonzero()[0]]
overlap_in_B = percent_overlap.iloc[percent_overlap["percent in B"].nonzero()[0]]

print(overlap_in_A.shape)
print(overlap_in_B.shape)
overlap_in_A


# In[16]:


overlap_in_B


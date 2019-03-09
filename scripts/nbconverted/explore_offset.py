
# coding: utf-8

# # Explore offset vector
# 
# We want to know what the offset vector is capturing.  Theoretically it should be capturing the "essence of gene A" since it is defined by taking the samples with the highest expression of gene A and the lowest expression of gene A.
# 
# We want to test if this offset vector is capturing genes in group A and B

# In[1]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

randomState = 123
from numpy.random import seed
seed(randomState)


# In[2]:


# Load data
base_dir = os.path.join(os.path.dirname(os.getcwd()), "data")
analysis_name = "sim_AB_2775_300_v2"
offset_gene_file = os.path.join(os.path.dirname(os.getcwd()), "data", analysis_name, "offset_gene_space.txt")
offset_vae_file = os.path.join(os.path.dirname(os.getcwd()), "encoded", analysis_name, "offset_latent_space_vae.txt")
A_file = os.path.join(base_dir, analysis_name, "geneSetA.txt")
B_file = os.path.join(base_dir, analysis_name, "geneSetB.txt")
weight_file = os.path.join(os.path.dirname(os.getcwd()), "data", analysis_name, "VAE_weight_matrix.txt")


# In[3]:


# Read gene space offset
offset_gene_space = pd.read_table(offset_gene_file, header=0, index_col=0)
offset_gene_space


# In[4]:


# Read VAE space offset
offset_vae_space = pd.read_table(offset_vae_file, header=0, index_col=0)
offset_vae_space


# In[5]:


# Read genes in set A
geneSetA = pd.read_table(A_file, header=0, index_col=0)
geneSetA_ls = [l[0] for l in geneSetA.values.tolist()]
geneSetA_set = set(geneSetA_ls)


# In[6]:


# Read genes in set B
geneSetB = pd.read_table(B_file, header=0, index_col=0)
geneSetB_ls = [l[0] for l in geneSetB.values.tolist()]
geneSetB_set = set(geneSetB_ls)


# In[7]:


# Read weight matrix
weight = pd.read_table(weight_file, header=0, index_col=0).T
weight.head(5)


# ## Explore gene space offset
# 
# 1.  What genes are most highly weighted?
# 2.  What percentage of these genes are in gene set A and B?

# In[8]:


# Distribution of weights in offset vector
sns.distplot(offset_gene_space)


# In[9]:


# Get gene ids with the highest weight from the offset vector
percentile = 95
threshold = np.percentile(offset_gene_space, percentile)
print("Threshold cutoff is {}".format(threshold))
highest_genes = offset_gene_space.T[offset_gene_space.T[0] > threshold].index


# In[10]:


# Compare the overlap of genes in set A and highest weighted genes in offset
venn2([set(highest_genes), geneSetA_set], set_labels = ('High weight offset genes', 'Group A genes'))
plt.show()


# In[11]:


# Compare the overlap of genes in set B and highest weighted genes in offset
venn2([set(highest_genes), geneSetB_set], set_labels = ('High weight offset genes', 'Group B genes'))
plt.show()


# ## Explore latent space (VAE) offset
# 1.  Which feature has the highest value?
# 2.  Are genes in set A and B highly weighted 

# In[12]:


# Distribution of weights in offset vector
sns.distplot(offset_vae_space)


# In[13]:


# Get latent feature with the max and min value
max_feature = offset_vae_space.T.idxmax()[0]
min_feature = offset_vae_space.T.idxmin()[0]
print("Max feature is {} and min feature is {}".format(max_feature, min_feature))


# ### Genes in feature that corresponds to max offset score

# In[14]:


# Get gene weights for max latent feature
genes_max_feature = weight[int(max_feature)]
sns.distplot(genes_max_feature)


# In[15]:


# Get gene ids with the highest positive weight from the max feature selected
percentile = 95
threshold = np.percentile(genes_max_feature, percentile)
print("Threshold cutoff is {}".format(threshold))
highest_genes = genes_max_feature[genes_max_feature > threshold].index


# In[16]:


# Get gene ids with the highest negative weight from the max feature selected
percentile = 5
threshold = np.percentile(genes_max_feature, percentile)
print("Threshold cutoff is {}".format(threshold))
lowest_genes = genes_max_feature[genes_max_feature < threshold].index


# In[17]:


# Compare the overlap of genes in set A and highest positive weighted genes in the max feature
venn2([set(highest_genes), geneSetA_set], set_labels = ('High positive weight genes in feature {}'.format(max_feature), 'Group A genes'))
plt.show()


# In[18]:


# Output intersected sets
intersect_highpos_geneA = geneSetA_set.intersection(set(highest_genes))
intersect_highpos_geneA_df = pd.DataFrame(list(intersect_highpos_geneA), columns=['gene id'])

intersect_file = os.path.join(base_dir, analysis_name, "intersect_feature{}_highpos_geneA.txt".format(max_feature))
intersect_highpos_geneA_df.to_csv(intersect_file, sep='\t')


# In[19]:


# Compare the overlap of genes in set B and highest positive weighted genes in the max feature
venn2([set(highest_genes), geneSetB_set], set_labels = ('High positive weight genes in feature {}'.format(max_feature), 'Group B genes'))
plt.show()


# In[20]:


# Output intersected sets
intersect_highpos_geneB = geneSetB_set.intersection(set(highest_genes))
intersect_highpos_geneB_df = pd.DataFrame(list(intersect_highpos_geneB), columns=['gene id'])

intersect_file = os.path.join(base_dir, analysis_name, "intersect_feature{}_highpos_geneB.txt".format(max_feature))
intersect_highpos_geneB_df.to_csv(intersect_file, sep='\t')


# In[21]:


# Compare the overlap of genes in set A and highest negative weighted genes in the max feature
venn2([set(lowest_genes), geneSetA_set], set_labels = ('High negative weight genes in feature {}'.format(max_feature), 'Group A genes'))
plt.show()


# In[22]:


# Compare the overlap of genes in set B and highest negative weighted genes in the max feature
venn2([set(lowest_genes), geneSetB_set], set_labels = ('High negative weight genes in feature {}'.format(max_feature), 'Group B genes'))
plt.show()


# In[23]:


# Output intersected sets
intersect_highneg_geneB = geneSetB_set.intersection(set(lowest_genes))
intersect_highneg_geneB_df = pd.DataFrame(list(intersect_highneg_geneB), columns=['gene id'])

intersect_file = os.path.join(base_dir, analysis_name, "intersect_feature{}_highneg_geneB.txt".format(max_feature))
intersect_highneg_geneB_df.to_csv(intersect_file, sep='\t')


# ### Genes in feature that corresponds to minimum offset score

# In[24]:


# Get gene weights for min latent feature
genes_min_feature = weight[int(min_feature)]
sns.distplot(genes_min_feature)


# In[25]:


# Get gene ids with the highest positive weight from the min feature selected
percentile = 95
threshold = np.percentile(genes_min_feature, percentile)
print("Threshold cutoff is {}".format(threshold))
highest_genes = genes_min_feature[genes_min_feature > threshold].index


# In[26]:


# Get gene ids with the highest negative weight from the min feature selected
percentile = 5
threshold = np.percentile(genes_min_feature, percentile)
print("Threshold cutoff is {}".format(threshold))
lowest_genes = genes_min_feature[genes_min_feature < threshold].index


# In[27]:


# Compare the overlap of genes in set A and highest positive weighted genes in the min feature
venn2([set(highest_genes), geneSetA_set], set_labels = ('High positive weight genes in feature {}'.format(min_feature), 'Group A genes'))
plt.show()


# In[28]:


# Compare the overlap of genes in set B and highest positive weighted genes in the min feature
venn2([set(highest_genes), geneSetB_set], set_labels = ('High positive weight genes in feature {}'.format(min_feature), 'Group B genes'))
plt.show()


# In[29]:


# Output intersected sets
intersect_highpos_geneB = geneSetB_set.intersection(set(highest_genes))
intersect_highpos_geneB_df = pd.DataFrame(list(intersect_highpos_geneB), columns=['gene id'])

intersect_file = os.path.join(base_dir, analysis_name, "intersect_feature{}_highpos_geneB.txt".format(min_feature))
intersect_highpos_geneB_df.to_csv(intersect_file, sep='\t')


# In[30]:


# Compare the overlap of genes in set A and highest negative weighted genes in the min feature
venn2([set(lowest_genes), geneSetA_set], set_labels = ('High negative weight genes in feature {}'.format(min_feature), 'Group A genes'))
plt.show()


# In[31]:


# Output intersected sets
intersect_highneg_geneA = geneSetA_set.intersection(set(lowest_genes))
intersect_highneg_geneA_df = pd.DataFrame(list(intersect_highneg_geneA), columns=['gene id'])

intersect_file = os.path.join(base_dir, analysis_name, "intersect_feature{}_highneg_geneA.txt".format(min_feature))
intersect_highneg_geneA_df.to_csv(intersect_file, sep='\t')


# In[32]:


# Compare the overlap of genes in set B and highest negative weighted genes in the min feature
venn2([set(lowest_genes), geneSetB_set], set_labels = ('High negative weight genes in feature {}'.format(min_feature), 'Group B genes'))
plt.show()


# In[33]:


# Output intersected sets
intersect_highneg_geneB = geneSetB_set.intersection(set(lowest_genes))
intersect_highneg_geneB_df = pd.DataFrame(list(intersect_highneg_geneB), columns=['gene id'])

intersect_file = os.path.join(base_dir, analysis_name, "intersect_feature{}_highneg_geneB.txt".format(min_feature))
intersect_highneg_geneB_df.to_csv(intersect_file, sep='\t')


# Observation:
# 
# Notice that the overlap of the high weight genes in the min features and max feature are very similar -- why is this?

# coding: utf-8

# # Simulation experiment
# 
# **Hypothesis**: If we generate a gene expression dataset with a strong signal, a nonlinear compression method (VAE)
#             will be able to learn this signal and predict new gene expression patterns better compared to 
#             using a linear comparession method (PCA) and using no compression method (all the genes)
#  
# **Study design**:
# 
# *(Input)* Add signal to Pa gene expression dataset:
# 
# Network: gene set A --> gene set B
# 
# Add signal to relate A and B using the following algorithm:
# 
# Hyperparmeters should include:
#  1. Size of gene set A
#  2. Size of gene set B
#  3. Effect size
#  4. Log file with hyperparameter selections
#          
# *(Approach)* Train nonlinear (VAE) and linear (PCA) compression algorithms using this simulated data
# 
# *(Evaluation)*  For each sample in the Pa dataset compare corr(predicted expression, actual expression)
# 
# *(Output)* Figure of the correlation scores per sample 

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import pandas as pd
import numpy as np
import random
import seaborn as sns

from functions import generate_input, vae, def_offset, interpolate, pca, plot

from numpy.random import seed
randomState = 123
seed(randomState)


# In[2]:


# Hyperparameters

# Size of the gene set A
geneSetA_size = 1000

# Size of the gene set that will be regulated by gene A
geneSetB_size = 1000 

# Set mean and standard deviation for distribution of A genes
mu_A, sigma_A = 0.5, 0.4 # mean and standard deviation

# Set mean and standard deviation for distribution of B genes if mean(A) is ABOVE threshold
mu_B, sigma_B = 0.8, 0.1 # mean and standard deviation

# Name of analysis
analysis_name = 'sim_distAB_1000AB'


# In[3]:


# Create list of base directories
base_dirs = [os.path.join(os.path.dirname(os.getcwd()), 'data'),
             os.path.join(os.path.dirname(os.getcwd()), 'encoded'),
             os.path.join(os.path.dirname(os.getcwd()), 'models'),
             os.path.join(os.path.dirname(os.getcwd()), 'output'),
             os.path.join(os.path.dirname(os.getcwd()), 'stats'),
             os.path.join(os.path.dirname(os.getcwd()), 'viz')    
]

# Check if directory exist otherwise create
for each_dir in base_dirs:
    analysis_dir = os.path.join(each_dir, analysis_name)
    
    if os.path.exists(analysis_dir):
        print('directory already exists: {}'.format(analysis_dir))
    else:
        print('creating new directory: {}'.format(analysis_dir))
    os.makedirs(analysis_dir, exist_ok=True)


# In[4]:


# Load arguments
data_file = os.path.join(os.path.dirname(os.getcwd()), "data", "all-pseudomonas-gene-normalized.zip")


# In[5]:


# Read in data
data = pd.read_table(data_file, header=0, sep='\t', index_col=0, compression='zip').T
data.head(5)


# In[6]:


# Randomly select genes for set A
gene_ids = list(data.columns)
geneSetA = random.sample(gene_ids, geneSetA_size)


# In[7]:


# checkpoint:  Check the number of genes
assert(len(gene_ids)==5549)
print("Confirmed that all gene ids are available")


# In[8]:


# Remove gene ids that were selected for gene set A
gene_ids = [elem for elem in gene_ids if elem not in geneSetA]
print("After removing {} gene ids for set A, there are {} gene ids remaining."
      .format(geneSetA_size, len(gene_ids)))


# In[9]:


# Randomly select genes for gene set B
geneSetB = random.sample(gene_ids, geneSetB_size)

# Remove gene ids that were selected for gene set B
gene_ids = [elem for elem in gene_ids if elem not in geneSetB]
print("After removing {} gene ids for set B, there are {} gene ids remaining."
      .format(geneSetB_size, len(gene_ids)))


# In[10]:


# checkpoint:  Check that genes in set A are distinct from genes in set B
assert(len(set(geneSetA).intersection(set(geneSetB))) == 0)


# ## Output gene groupings 
# Output gene assignments (group A, B) to be used in [explore_simulated_data.py](explore_simulated_data.ipynb)

# In[11]:


# Output gene groupings
geneSetA_df = pd.DataFrame(geneSetA, columns=['gene id'])
geneSetB_df = pd.DataFrame(geneSetB, columns=['gene id'])

geneSetA_file = os.path.join(os.path.dirname(os.getcwd()), "data", analysis_name, "geneSetA.txt")
geneSetB_file = os.path.join(os.path.dirname(os.getcwd()), "data", analysis_name, "geneSetB.txt")


geneSetA_df.to_csv(geneSetA_file, sep='\t')
geneSetB_df.to_csv(geneSetB_file, sep='\t')


# In[12]:


data[geneSetA]


# In[13]:


data[geneSetB]


# ## Add artificial signal to the data
# Algorithm:
# ```python
# for sample in compendium:
#     expression(gene in set_A) is sampled from Normal(meanA,stdA)
#     if mean(all genes in set A) > meanA:
#         expression(gene in set B) sampled from Normal(meanB1, stdB1)
#     else
#         expression(gene in set B) sampled from Normal(meanB2, stdB2) 
# ```
# Note: This algorithm is applied to each sample in the compendium 
#       (essentially adding a signal to the existing gene expression data in the compendium)

# In[14]:


# Number of samples 
num_samples = data.shape[1]

# Loop through all samples
for sample_id in data.index:
    
    # Distribution of genes in set A and B using user params
    geneA_dist = np.random.normal(mu_A, sigma_A, geneSetA_size)
    geneB_dist = np.random.normal(mu_B, sigma_B, geneSetB_size)
    
    # Set gene expression value for genes in set A to be sampled from geneA_dist
    sample_A = data.loc[sample_id][geneSetA]
    data.loc[sample_id][geneSetA] = pd.Series(geneA_dist, index=sample_A.index)
    
    # Check if expression of genes in set A exceed the threshold
    # Use representatve gene from set A "geneSetA_pick" selected above
    if data.loc[sample_id,geneSetA].mean() > mu_A:
        
        # Sample gene B from geneB_dist_above
        sample_B = data.loc[sample_id][geneSetB]
        data.loc[sample_id][geneSetB] = pd.Series(geneB_dist, index=sample_B.index)            
            
# if any exceed 1 then set to 1 since gene expression is normalized
data[data>=1.0] = 1.0

# if any below 0 then set to 0 since gene expression is normalized
data[data<0.0] = 0.0


# In[15]:


data[geneSetA]


# In[16]:


data[geneSetB]


# ## Outputs
# Output
# 1. Simulated dataset (\data)
# 2. Log file containing hyperparmeters used (\metadata)
# 3. Expression of representative gene from group A (\data)

# In[17]:


# Output the new gene expression values for each sample
train_input_file = os.path.join(base_dirs[0], analysis_name, "train_model_input.txt.xz")

# Only include genes in group A and B
geneSetAB = geneSetA + geneSetB
simplified_data = data[geneSetAB]
simplified_data.to_csv(train_input_file, sep='\t', compression='xz', float_format="%.5g")

# Output log file with parameters used to generate simulated data
log_file = os.path.join(os.path.dirname(os.getcwd()), 'metadata', analysis_name + '_log.txt')

args_dict = {
    "size of gene set A": geneSetA_size,
    "size of gene set B": geneSetB_size,
    "mean of gene set A": mu_A,
    "std of gene set A": sigma_A,
    "mean of gene set B": mu_B,
    "std of gene set B": sigma_B
}

with open(log_file, 'w') as f:
    for key, value in args_dict.items():
        f.write('%s: %s\n' % (key, value))


# ## Train 
# Train compression methods (VAE, PCA) using simulated data

# In[18]:


get_ipython().run_cell_magic('time', '', '# Parameters to train nonlinear (VAE) compression method\nlearning_rate = 0.001\nbatch_size = 100\nepochs = 100\nkappa = 0.01\nintermediate_dim = 100\nlatent_dim = 2\nepsilon_std = 1.0\nnum_PCs = latent_dim\n\nbase_dir = os.path.dirname(os.getcwd())\n\n# Train nonlinear (VAE)\nvae.tybalt_2layer_model(learning_rate, batch_size, epochs, kappa, intermediate_dim,\n                        latent_dim, epsilon_std, base_dir, analysis_name)\n# Train linear (PCA)\npca.pca_model(base_dir, analysis_name, num_PCs)')


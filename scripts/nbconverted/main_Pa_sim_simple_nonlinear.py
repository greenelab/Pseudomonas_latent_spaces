
# coding: utf-8

# In[1]:


#-----------------------------------------------------------------------------------------------------------------
# Simulation experiment
#
# Network:
# A --> gene set C
#
# Algorithm
# Let gene A and B be TF's:
# if gene A expression > threshold_A:
#     genes in set C are upregulated by some percentage of its original expression
#
# Apply this algorithm for each sample in the compendium (essentially adding a signal to the existing gene expression data in the compendium)
# 
# Hyperparmeters should include: 
# 1. Gene A
# 2. Size of gene sets C
# 3. Thresholds
# 4. Percentage to 1.0 (effect size)
# 5. Log file with hyperparameter selections
#-----------------------------------------------------------------------------------------------------------------
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import pandas as pd
import numpy as np
import random

from functions import generate_input, vae, def_offset, interpolate, pca, plot

from numpy.random import seed
randomState = 5
seed(randomState)


# In[2]:


# Hyperparameters

# Transcription factors
# If empty, genes will be randomly assigned
geneA = ''

# Size of the gene set that will be regulated by gene A
geneSetC_size = 1000 

# Threshold for activation of gene A 
thresholdA = 0.5

# Amount that genes in gene sets C 
effect_sizeA = 0.5

# Name of analysis
analysis_name = 'sim_nonlin_5'


# In[3]:


# Load arguments
data_file = os.path.join(os.path.dirname(os.getcwd()), "data", "all-pseudomonas-gene-normalized.zip")


# In[4]:


# Read in data
data = pd.read_table(data_file, header=0, sep='\t', index_col=0, compression='zip').T
data.head(5)


# In[5]:


# Randomly select gene A if not specified
# Note: 'replace=False' indicates sampling WITHOUT replacement
if not geneA:
    gene_ids = list(data.columns)
    [geneA] = np.random.choice(gene_ids, size=1, replace=False)

print(geneA)


# In[6]:


# checkpoint
assert(len(gene_ids)==5549)


# In[7]:


# Randomly select genes for gene set C
# remove() doesn't return a value it will remove the element from the list object
gene_ids.remove(geneA)
print(len(gene_ids))

# Random sample of genes for set C
geneSetC = random.sample(gene_ids, geneSetC_size)

print(len(geneSetC))


# In[8]:


# checkpoint
assert(geneA not in geneSetC)


# In[9]:


# checkpoint
# print(data[geneA])


# In[10]:


# checkpoint: before transformation
# data.loc[data[geneA]<=thresholdA,geneSetC[0]]


# In[11]:


# Loop through all samples
num_samples = data.shape[1]

for sample_id in data.index:
    row = data.loc[sample_id]
        
    if data.loc[sample_id,geneA] > thresholdA:
        # Scale genes by some fixed percentage
        for gene in geneSetC:
            data.loc[sample_id,gene] = data.loc[sample_id,gene]*(1+effect_sizeA)            
            
# if any exceed 1 then set to 1 since gene expression is normalized
data[data>=1.0] = 1.0


# In[12]:


# checkpoint: after transformation
# data.loc[data[geneA]<=thresholdA,geneSetC[0]]


# In[13]:


# Dataframe with only gene A
geneA_only = pd.DataFrame(data[geneA], index=data.index, columns=[geneA])

# Drop gene A
data_holdout = data.drop(columns=[geneA])


# In[14]:


# checkpoint
data_holdout.shape


# In[15]:


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
        os.mkdir(analysis_dir)
        print('creating new directory: {}'.format(analysis_dir))


# In[16]:


# Output the new gene expression values for each sample
train_input_file = os.path.join(base_dirs[0], analysis_name, "train_model_input.txt.xz")
data_holdout.to_csv(train_input_file, sep='\t', compression='xz', float_format="%.5g")

# Output log file with params
log_file = os.path.join(os.path.dirname(os.getcwd()), 'metadata', analysis_name)

args_dict = {
    "gene A": geneA,
    "size of gene set C": geneSetC_size,
    "genes in set C": geneSetC,
    "threshold of A activation": thresholdA,
    "effect size of genes in set C": effect_sizeA,
}

with open(log_file, 'w') as f:
    for key, value in args_dict.items():
        f.write('%s: %s\n' % (key, value))
        
# Output geneA only file
geneA_file = os.path.join(base_dirs[0], analysis_name, geneA + ".txt")
geneA_only.to_csv(geneA_file, sep='\t', float_format="%.5g")


# In[17]:


get_ipython().run_line_magic('time', '')
# Train models on input dataset

# params
learning_rate = 0.001
batch_size = 100
epochs = 200
kappa = 0.01
intermediate_dim = 100
latent_dim = 10
epsilon_std = 1.0
num_PCs = latent_dim

base_dir = os.path.dirname(os.getcwd())
vae.tybalt_2layer_model(learning_rate, batch_size, epochs, kappa, intermediate_dim, latent_dim, epsilon_std, base_dir, analysis_name)
pca.pca_model(base_dir, analysis_name, num_PCs)


# Define offset vectors in gene space
data_dir = os.path.join(base_dirs[0], analysis_name)
target_gene = geneA
percent_low = 5
percent_high = 95

def_offset.gene_space_offset(data_dir, target_gene, percent_low, percent_high)


# Define offset vectors for different latent spaces
model_dir = os.path.join(base_dirs[2], analysis_name)
encoded_dir = os.path.join(base_dirs[1], analysis_name)

def_offset.vae_latent_space_offset(data_dir, model_dir, encoded_dir, target_gene, percent_low, percent_high)
def_offset.pca_latent_space_offset(data_dir, model_dir, encoded_dir, target_gene, percent_low, percent_high)


# Predict gene expression using offset in gene space and latent space
out_dir = os.path.join(base_dirs[3], analysis_name)

interpolate.interpolate_in_gene_space(data_dir, target_gene, out_dir, percent_low, percent_high)
interpolate.interpolate_in_vae_latent_space(data_dir, model_dir, encoded_dir, target_gene, out_dir, percent_low, percent_high)
interpolate.interpolate_in_pca_latent_space(data_dir, model_dir, encoded_dir, target_gene, out_dir, percent_low, percent_high)


# Plot prediction per sample along gradient of PA1673 expression
viz_dir = os.path.join(base_dirs[5], analysis_name)
plot.plot_corr_gradient(out_dir, viz_dir)


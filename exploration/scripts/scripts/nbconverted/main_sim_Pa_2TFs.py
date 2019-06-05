
# coding: utf-8

# In[1]:


#-----------------------------------------------------------------------------------------------------------------
# Simulation experiment
#
# Network:
# A--> B --> gene set D
# A --> gene set C
#
# Algorithm
# Let gene A and B be TF's:
# if gene A expression > threshold_A:
#     genes in set C are upregulated
#     gene B is upregulated
# if gene B expression > threshold_B:
#    genes in set D are upregulated
#
# Apply this algorithm for each sample in the compendium (essentially adding a signal to the existing gene expression data in the compendium)
# 
# Hyperparmeters should include: 
# 1. Gene A, B
# 2. Size of gene sets C and D (D << C)
# 3. Thresholds
# 4. Percentage to 1.0 (effect size)
# 5. Gene set C and D mutually exclusive?
# 6. Log file with hyperparameter selections
#-----------------------------------------------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import random

from functions import generate_input, vae, def_offset, interpolate, plot

from numpy.random import seed
randomState = 123
seed(randomState)


# In[2]:


# Hyperparameters

# Transcription factors
# If empty, genes will be randomly assigned
geneA = ''
geneB = ''

# Size of the gene sets that will be regulated by gene A and B
# Note gene set C will automatically include gene B based on network architecture
geneSetC_size = 100 
geneSetD_size = 100

# Percentage of mixture between the gene sets C and D (i.e. percentabe of genes that overlap)
mixture = 0

# Threshold for activation of gene A and gene B
thresholdA = 0.5
thresholdB = 0.7

# Amount that genes in gene sets C and D are upregulated
effect_sizeA = 0.3
effect_sizeB = 0.3

# Name of analysis
analysis_name = 'sim_1_8'


# In[3]:


# Load arguments
data_file = os.path.join(os.path.dirname(os.getcwd()), "data", "all-pseudomonas-gene-normalized.zip")


# In[4]:


# Read in data
data = pd.read_table(data_file, header=0, sep='\t', index_col=0, compression='zip').T
data.head(5)


# In[5]:


# Randomly select gene A and gene B if not specified
# Note: 'replace=False' indicates sampling WITHOUT replacement
if not geneA and not geneB:
    gene_ids = list(data.columns)
    [geneA, geneB] = np.random.choice(gene_ids, size=2, replace=False)

print(geneA)
print(geneB)


# In[6]:


# Randomly select genes for gene set C
# remove() doesn't return a value it will remove the element from the list object
gene_ids.remove(geneA)
gene_ids.remove(geneB)
print(len(gene_ids))

# Random sample of genes for set C
geneSetC = random.sample(gene_ids, geneSetC_size)

# Remove selected genes
gene_ids = [c for c in gene_ids if c not in geneSetC]

print(len(geneSetC))


# In[7]:


# Randomly select genes for gene set D, accounting for percent mixture with gene set C
#geneSetD = random.sample(gene_ids, geneSetD_size)

# Sample a percentage of genes from gene set C
num_shared = round(mixture*geneSetC_size)
num_remaining = geneSetD_size - num_shared

# Add sampled percentage of genes to gene set D
# Fill in remaining samples for gene set D
geneSetD = random.sample(geneSetC, num_shared)
geneSetD = geneSetD + random.sample(gene_ids, num_remaining)

print(len(geneSetD))


# In[8]:


# Add geneB to gene set C
geneSetC.append(geneB)


# In[9]:


# Loop through all samples
num_samples = data.shape[1]

for sample_id in data.index:
    row = data.loc[sample_id]
        
    if data.loc[sample_id,geneA] > thresholdA:
        # Scale genes by some fixed percentage
        for gene in geneSetC:
            data.loc[sample_id,gene] = data.loc[sample_id,gene]*(1+effect_sizeA)            
    if data.loc[sample_id,geneB] > thresholdB:
        for gene in geneSetD:
            data.loc[sample_id,gene] = data.loc[sample_id,gene]*(1+effect_sizeB)
            
# if any exceed 1 then set to 1 since gene expression is normalized
data[data>=1.0] = 1.0


# In[10]:


# Dataframe with only gene A
geneA_only = pd.DataFrame(data[geneA], index=data.index, columns=[geneA])

# Drop gene A
data_holdout = data.drop(columns=[geneA])


# In[11]:


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


# In[12]:


# Output the new gene expression values for each sample
train_input_file = os.path.join(base_dirs[0], analysis_name, "train_model_input.txt.xz")
data_holdout.to_csv(train_input_file, sep='\t', compression='xz', float_format="%.5g")

# Output log file with params
log_file = os.path.join(os.path.dirname(os.getcwd()), 'metadata', analysis_name + '.txt')

args_dict = {
    "gene A": geneA,
    "gene B": geneB,
    "size of gene set C": geneSetC_size,
    "genes in set C": geneSetC,
    "size of gene set D": geneSetD_size,
    "genes in set D": geneSetD,
    "percentage of shared genes between C and D": mixture,
    "threshold of A activation": thresholdA,
    "threshold of B activation": thresholdB,
    "effect size of genes in set C": effect_sizeA,
    "effect size of genes in set D": effect_sizeB
}

with open(log_file, 'w') as f:
    for key, value in args_dict.items():
        f.write('%s: %s\n' % (key, value))
        
# Output geneA only file
geneA_file = os.path.join(base_dirs[0], analysis_name, geneA + ".txt")
geneA_only.to_csv(geneA_file, sep='\t', float_format="%.5g")


# In[13]:


get_ipython().run_line_magic('time', '')
# Run Tybalt
learning_rate = 0.001
batch_size = 100
epochs = 200
kappa = 0.01
intermediate_dim = 100
latent_dim = 10
epsilon_std = 1.0

base_dir = os.path.dirname(os.getcwd())
vae.tybalt_2layer_model(learning_rate, batch_size, epochs, kappa, intermediate_dim, latent_dim, epsilon_std, base_dir, analysis_name)


# Define offset vectors in gene space and latent space
data_dir = os.path.join(base_dirs[0], analysis_name)
target_gene = geneA
percent_low = 5
percent_high = 95

def_offset.gene_space_offset(data_dir, target_gene, percent_low, percent_high)

model_dir = os.path.join(base_dirs[2], analysis_name)
encoded_dir = os.path.join(base_dirs[1], analysis_name)

def_offset.latent_space_offset(data_dir, model_dir, encoded_dir, target_gene, percent_low, percent_high)


# Predict gene expression using offset in gene space and latent space
out_dir = os.path.join(base_dirs[3], analysis_name)

interpolate.interpolate_in_gene_space(data_dir, target_gene, out_dir, percent_low, percent_high)
interpolate.interpolate_in_latent_space(data_dir, model_dir, encoded_dir, target_gene, out_dir, percent_low, percent_high)


# Plot prediction per sample along gradient of PA1673 expression
viz_dir = os.path.join(base_dirs[5], analysis_name)
plot.plot_corr_gradient(out_dir, viz_dir)


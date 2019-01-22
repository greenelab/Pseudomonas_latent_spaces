
# coding: utf-8

# In[1]:


#-----------------------------------------------------------------------------------------------------------------
# Simulation experiment
#
# Network:
# gene set A --> gene set B
#
# Algorithm
# Let gene set A be TF's:
# if genes in set A expression > threshold_A:
#     genes in set B are set to some proportion of the expression of gene A
#
# Apply this algorithm for each sample in the compendium (essentially adding a signal to the existing gene expression data in the compendium)
# 
# Hyperparmeters should include: 
# 1. Size of gene set A
# 2. Size of gene set B
# 3. Proportion of gene A expression
# 4. Thresholds
# 5. Log file with hyperparameter selections
#-----------------------------------------------------------------------------------------------------------------
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
geneSetA_size = 100

# Size of the gene set that will be regulated by gene A
geneSetB_size = 1000 

# Percentage to upregulate each gene in set B
effect_size = 0.5

# Threshold for activation of gene A 
thresholdA = 0.5

# Name of analysis
analysis_name = 'sim_AB_2775_300_skewA'


# In[3]:


# Load arguments
data_file = os.path.join(os.path.dirname(os.getcwd()), "data", "all-pseudomonas-gene-normalized.zip")


# In[4]:


# Read in data
data = pd.read_table(data_file, header=0, sep='\t', index_col=0, compression='zip').T
data.head(5)


# In[5]:


# Randomly select genes for set A
gene_ids = list(data.columns)
geneSetA = random.sample(gene_ids, geneSetA_size)


# In[6]:


# checkpoint:  Check the number of genes
assert(len(gene_ids)==5549)
print("Confirmed that all gene ids are available")


# In[7]:


# Remove gene ids that were selected for gene set A
gene_ids = [elem for elem in gene_ids if elem not in geneSetA]
print("After removing {} gene ids for set A, there are {} gene ids remaining.".format(geneSetA_size, len(gene_ids)))


# In[8]:


# Randomly select genes for gene set B
geneSetB = random.sample(gene_ids, geneSetB_size)

# Remove gene ids that were selected for gene set B
gene_ids = [elem for elem in gene_ids if elem not in geneSetB]
print("After removing {} gene ids for set B, there are {} gene ids remaining.".format(geneSetB_size, len(gene_ids)))


# In[9]:


# checkpoint:  Check that genes in set A are distinct from genes in set B
assert(len(set(geneSetA).intersection(set(geneSetB))) == 0)


# In[10]:


# Main input data simulation 

# Loop through all samples
num_samples = data.shape[1]
for sample_id in data.index:
    row = data.loc[sample_id]
    
    # Randomly select a value [0,1] where each value is uniformly likely to be chosen
    new_A_exp = random.uniform(0.4, 1.0)
    
    # Set gene expression value for genes in set A to be the same random value selected
    data.loc[sample_id][geneSetA] = new_A_exp
    
    # Select sample gene from set A to be representative since the expression is the same
    # for all genes in the set
    geneSetA_pick = geneSetA[0]
    
    # Check if expression of genes in set A exceed the threshold
    if data.loc[sample_id,geneSetA_pick] > thresholdA:
        
        # Scale genes by some fixed percentage
        for gene in geneSetB:
            data.loc[sample_id,gene] = (1+effect_size)*data.loc[sample_id,gene]            
            
# if any exceed 1 then set to 1 since gene expression is normalized
data[data>=1.0] = 1.0


# In[11]:


# Dataframe with only genes in set A
geneA_only = pd.DataFrame(data[geneSetA_pick], index=data.index, columns=[geneSetA_pick])

# Drop genes in set A
data_holdout = data.drop(geneSetA, axis=1)

geneA_only


# In[12]:


# checkpoint:  Check that the holdout set does not include genes in set A
assert(data_holdout.shape[1] == (data.shape[1]-len(geneSetA)))
print("Confirmed that holdout set is the correct size")


# In[13]:


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


# In[14]:


# Output the new gene expression values for each sample
train_input_file = os.path.join(base_dirs[0], analysis_name, "train_model_input.txt.xz")
data_holdout.to_csv(train_input_file, sep='\t', compression='xz', float_format="%.5g")

# Output log file with params
log_file = os.path.join(os.path.dirname(os.getcwd()), 'metadata', analysis_name + '.txt')

args_dict = {
    "size of gene set A": geneSetA_size,
    "size of gene set B": geneSetB_size,
    "genes in set B": geneSetB,
    "effect size of genes in set B": effect_size,
    "threshold of A activation": thresholdA,
}

with open(log_file, 'w') as f:
    for key, value in args_dict.items():
        f.write('%s: %s\n' % (key, value))
        
# Output geneA only file using sample gene A selected
geneA_file = os.path.join(base_dirs[0], analysis_name, geneSetA_pick + ".txt")
geneA_only.to_csv(geneA_file, sep='\t', float_format="%.5g")


# In[15]:


get_ipython().run_cell_magic('time', '', '# Train models on input dataset\n\n# params\nlearning_rate = 0.001\nbatch_size = 100\nepochs = 200\nkappa = 0.01\nintermediate_dim = 2775\nlatent_dim = 300\nepsilon_std = 1.0\nnum_PCs = latent_dim\n\nbase_dir = os.path.dirname(os.getcwd())\nvae.tybalt_2layer_model(learning_rate, batch_size, epochs, kappa, intermediate_dim,\n                        latent_dim, epsilon_std, base_dir, analysis_name)\npca.pca_model(base_dir, analysis_name, num_PCs)\n\n\n# Define offset vectors in gene space\ndata_dir = os.path.join(base_dirs[0], analysis_name)\ntarget_gene = geneSetA_pick\npercent_low = 5\npercent_high = 95\n\ndef_offset.gene_space_offset(data_dir, target_gene, percent_low, percent_high)\n\n\n# Define offset vectors for different latent spaces\nmodel_dir = os.path.join(base_dirs[2], analysis_name)\nencoded_dir = os.path.join(base_dirs[1], analysis_name)\n\ndef_offset.vae_latent_space_offset(data_dir, model_dir, encoded_dir, latent_dim, target_gene, percent_low, percent_high)\ndef_offset.pca_latent_space_offset(data_dir, model_dir, encoded_dir, target_gene, percent_low, percent_high)\n\n\n# Predict gene expression using offset in gene space and latent space\nout_dir = os.path.join(base_dirs[3], analysis_name)\n\ninterpolate.interpolate_in_gene_space(data_dir, target_gene, out_dir, percent_low, percent_high)\ninterpolate.interpolate_in_vae_latent_space(data_dir, model_dir, encoded_dir, latent_dim, \n                                            target_gene, out_dir, percent_low, percent_high)\ninterpolate.interpolate_in_pca_latent_space(data_dir, model_dir, encoded_dir, target_gene, out_dir, percent_low, percent_high)\n\n# True if the x-axis of the plot uses the sample index\n# False if the x-asix of the plot uses the gene expression of the target gene\nby_sample_ind = False\n\n# Plot prediction per sample along gradient of PA1673 expression\nviz_dir = os.path.join(base_dirs[5], analysis_name)\nplot.plot_corr_gradient(out_dir, viz_dir, target_gene, by_sample_ind)')



# coding: utf-8

# In[1]:


#-------------------------------------------------------------------------------------------------------------------------------
# By Alexandra Lee
# (updated October 2018)
# 
# Main
#
# Dataset: Pseudomonas aeruginosa gene expression from compendium 
# referenced in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5069748/
# 
# Condition: expression of PA1673 gene
#
# Task: To predict the expression of other (non-PA1673) genes by:
#        1. Define offset vector = avg(expression of genes corresponding to high levels of PA1673) 
#           - avg(expression of genes corresponding to low levels of PA1673)
#        2. scale factor = how far along the gradient of low-high PA1673 expression
#        3. prediction = baseline expression + scale factor * offset vector 
#-------------------------------------------------------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np

from functions import generate_input, vae, def_offset, interpolate, plot

randomState = 123
from numpy.random import seed
seed(randomState)


# In[2]:


# Name of analysis
analysis_name = 'PA1673_gradient_test'

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


# In[3]:


# Pre-process input
data_dir = os.path.join(base_dirs[0], analysis_name)
generate_input.generate_input_PA1673_gradient(data_dir)

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
target_gene = "PA1673"
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


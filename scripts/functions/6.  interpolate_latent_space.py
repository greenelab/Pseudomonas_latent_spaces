
# coding: utf-8

# In[1]:


#-------------------------------------------------------------------------------------------------------------------------------
# By Alexandra Lee (October 2018) 
#
# Interpolate in gene space
# Add scale factor of offset vector to each sample to transform the gene expression 
# profile along some gradient
#  
#-------------------------------------------------------------------------------------------------------------------------------
import os
import sys
import pandas as pd
import numpy as np
from keras.models import model_from_json, load_model
from keras import metrics, optimizers
from scipy.stats import pearsonr, spearmanr

sys.path.append("/home/Documents/Repos/Pseudomonas_latent_spaces/scripts/nbconverted/")

from nbconverted import utils

randomState = 123
from numpy.random import seed
seed(randomState)


# In[2]:


# Load arguments
PA1673_file = os.path.join(os.path.dirname(os.getcwd()), "data", "PA1673_gradient", "PA1673.txt")
nonPA1673_file = os.path.join(os.path.dirname(os.getcwd()), "data", "PA1673_gradient", "train_model_input.txt.xz")
offset_file = os.path.join(os.path.dirname(os.getcwd()), "encoded", "PA1673_gradient", "offset_latent_space.txt")

model_file = os.path.join(os.path.dirname(os.getcwd()), "models", "PA1673_gradient", "tybalt_2layer_10latent_encoder_model.h5")
weights_file = os.path.join(os.path.dirname(os.getcwd()), "models", "PA1673_gradient", "tybalt_2layer_10latent_encoder_weights.h5")
model_decode_file = os.path.join(os.path.dirname(os.getcwd()), "models", "PA1673_gradient", "tybalt_2layer_10latent_decoder_model.h5")
weights_decode_file = os.path.join(os.path.dirname(os.getcwd()), "models", "PA1673_gradient", "tybalt_2layer_10latent_decoder_weights.h5")

gene_id = 'PA1673'

# Output files
corr_file = os.path.join(os.path.dirname(os.getcwd()), "output", "PA1673_gradient", "corr_latent_space.txt")


# In[3]:


# Read in data
PA1673_data = pd.read_table(PA1673_file, header=0, sep='\t', index_col=0)
nonPA1673_data = pd.read_table(nonPA1673_file, header=0, sep='\t', index_col=0)
offset_encoded = pd.read_table(offset_file, header=0, sep='\t', index_col=0)

offset_encoded


# In[4]:


# read in saved models
loaded_model = load_model(model_file)
loaded_decode_model = load_model(model_decode_file)

# load weights into new model
loaded_model.load_weights(weights_file)
loaded_decode_model.load_weights(weights_decode_file)


# In[5]:


# Sort PA1673_data by expression (lowest --> highest)
PA1673_sorted = PA1673_data.sort_values(by=[gene_id])

# Get sample IDs with the lowest 5% of PA1673 expression
threshold_low = np.percentile(PA1673_sorted[gene_id], 5)
low_ids = PA1673_sorted[PA1673_sorted[gene_id]<= threshold_low].index
low_exp = nonPA1673_data.loc[low_ids]

# Use trained model to encode expression data into SAME latent space
low_exp_encoded = loaded_model.predict_on_batch(low_exp)
low_exp_encoded_df = pd.DataFrame(low_exp_encoded, index=low_exp.index)

# Average gene expression across samples in each extreme group
lowest_mean_encoded = low_exp_encoded.mean(axis=0)

# Format and rename as "baseline"
baseline_encoded = pd.DataFrame(lowest_mean_encoded, index=offset_encoded.columns).T
baseline_encoded


# In[6]:


# Loop through all samples in the compendium in order of the PA1673 expression
remaining_ids = PA1673_sorted[PA1673_sorted[gene_id]> threshold_low].index

corr_score = {}
for sample_id in remaining_ids:
    intermediate_PA1673_exp = PA1673_sorted.loc[sample_id]
    alpha = utils.get_scale_factor(PA1673_file, gene_id, intermediate_PA1673_exp)

    predict = baseline_encoded + alpha.values[0]*offset_encoded

    # Decode prediction
    predict_decoded = loaded_decode_model.predict_on_batch(predict)
    predict = pd.DataFrame(predict_decoded, columns=nonPA1673_data.columns)

    true = pd.Series.to_frame(nonPA1673_data.loc[sample_id]).T
    
    [coeff, pval] = pearsonr(predict.values.T, true.values.T)
    corr_score[sample_id] = coeff 
    
corr_score_df = pd.DataFrame.from_dict(corr_score, orient='index')


# In[7]:


# Output estimated gene experession values
corr_score_df.to_csv(corr_file, sep='\t')


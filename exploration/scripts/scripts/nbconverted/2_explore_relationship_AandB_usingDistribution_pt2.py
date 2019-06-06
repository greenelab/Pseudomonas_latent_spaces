
# coding: utf-8

# # Explore simulated relationship (part 2)
# 
# This notebook is using simulated data generated from [main_Pa_sim_enhance_AtoB](1_main_Pa_sim_enhance_AtoB.ipynb).  This notebook input raw Pseudomonas gene expression data from the Pseudomonas compendium referenced in [ADAGE](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5069748/) paper and added a strong nonlinear signal.  This signal assigned a set of genes to group A and a set of genes to group B.  If the expression of genes in group A exceeded some threshold then the genes in group B were upregulated.  
# 
# This notebook is extending from the exploration performed in [explore_relationship_AandB_pt1](explore_relationship_AandB_pt1.ipynb).  In this notebook we determined that the modeled/predicted gene expression data between A and B (i.e. after applying a linear transformation in the latent space and decoding) is a mostly linear relationship.  We assume that this means that the *decoder* is learning this linear relationship.  So now we want to determine what the *encoder* is learning. 

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import pandas as pd
import numpy as np
import random
import glob
import seaborn as sns
from keras.models import model_from_json, load_model
from functions import utils
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

from numpy.random import seed
randomState = 123
seed(randomState)


# In[2]:


# Run notebook to generate simulated data
#%run ./main_Pa_sim_enhance_AtoB.ipynb


# In[3]:


# Load 
base_dir = os.path.dirname(os.getcwd())
analysis_name = 'sim_distAB_1000AB'

sim_data_file = os.path.join(
    base_dir,
    "data",
    analysis_name,
    "train_model_input.txt.xz"
)

A_file = os.path.join(
    base_dir,
    "data",
    analysis_name,
    "geneSetA.txt"
)

B_file = os.path.join(
    base_dir,
    "data",
    analysis_name,
    "geneSetB.txt"
)

offset_vae_file = os.path.join(
    os.path.dirname(os.getcwd()), 
    "encoded",
    analysis_name, 
    "offset_latent_space_vae.txt"
)

weight_file = os.path.join(
    os.path.dirname(os.getcwd()),
    "data",
    analysis_name,
    "VAE_weight_matrix.txt"
)

model_encoder_file = glob.glob(os.path.join(
    base_dir,
    "models",
    analysis_name,
    "*_encoder_model.h5"))[0]

weights_encoder_file = glob.glob(
    os.path.join(
        base_dir,
        "models",
        analysis_name,
        "*_encoder_weights.h5"
    )
)[0]

model_decoder_file = glob.glob(
    os.path.join(
        base_dir,
        "models",
        analysis_name, 
        "*_decoder_model.h5"
    )
)[0]

weights_decoder_file = glob.glob(
    os.path.join(
        base_dir,
        "models", 
        analysis_name, 
        "*_decoder_weights.h5"
    )
)[0]


# In[4]:


# Read data
sim_data = pd.read_table(sim_data_file, index_col=0, header=0, compression='xz')
geneSetA = pd.read_table(A_file, header=0, index_col=0)
geneSetB = pd.read_table(B_file, header=0, index_col=0)

print(sim_data.shape)
sim_data.head()


# ## 1. Trend of gene B with respect to A (input)
# 
# How is B changing with respect to A in our simulated dataset (before the data goes into the autoencoder)?
# 
# Plot gene expression of A vs mean(gene B expression).  This plot will serve as a reference against later plots that will show the relationship between A and B after transforming the data (i.e. after the data has been fed through the autoencoder)

# In[5]:


# Get the means of ORIGINAL B genes

# Convert dataframe with gene ids to list
geneSetB_ls = geneSetB['gene id'].values.tolist()

geneSetB_exp = sim_data[geneSetB_ls]

# Get the mean for each sample
geneSetB_mean_original_all = geneSetB_exp.mean(axis=1)
geneSetB_mean_original_all.head()


# In[6]:


# Get the means of ORIGINAL A genes

# Convert dataframe with gene ids to list
geneSetA_ls = geneSetA['gene id'].values.tolist()

geneSetA_exp = sim_data[geneSetA_ls]

# Get the mean for each sample
geneSetA_mean_original_all = geneSetA_exp.mean(axis=1)
geneSetA_mean_original_all.head()


# In[7]:


# Join original expression of A and mean(transformed expression of B)
original_A_mean_exp = geneSetA_mean_original_all
original_B_mean_exp = geneSetB_mean_original_all

original_A_vs_original_B_df = pd.merge(original_A_mean_exp.to_frame('mean gene A untransformed'),
                      original_B_mean_exp.to_frame('mean gene B untransformed'),
                      left_index=True, right_index=True)
original_A_vs_original_B_df.head()


# **Plot**
# The plot below shows the signal that was added to the dataset.  This signal assigned a set of genes to group A and a set of genes to group B. If the expression of genes in group A exceeded some threshold then the genes in group B were upregulated.  
# 
# So we see a step function relationship between the expression of genes in group A and the expression of genes in group B.  With a threshold of 0.5 we can see that the expression of genes in B are upregulated.

# In[8]:


# Plot
sns.regplot(x='mean gene A untransformed',
            y='mean gene B untransformed',
           data = original_A_vs_original_B_df)


# ## 3.  Trend of gene B with respect to A (encoder)
# 
# How is B changing with respect to A after shifting input expression and then applying our latent space transformation?
# 
# Here we are only changing samples **before** they have been encoded into the latent space and then we apply our latent space transformation.  If we compare these trends with those from #2 module, which show what the decoder is supposedly learning, then we can conclude what the encoder is learning.
# 
# In order to test this we manually shift A genes from being below the activation threshold to being above it and see how the gene expression data is reconstructed

# In[9]:


# Artificially shift gene A expression

# Get single sample
test_sample = geneSetA_mean_original_all[geneSetA_mean_original_all == min(geneSetA_mean_original_all)].index[0]
print(test_sample)

# Sample with original value of gene A
A_exp_sample = sim_data.loc[test_sample]

A_exp_sample_modified_df = pd.DataFrame()

A_exp_sample_modified_df.append(A_exp_sample, ignore_index=True)

# Convert dataframe with gene ids to list
geneSetA_ls = geneSetA['gene id'].values.tolist()

# Artificially shift genes in set A
mu_As = np.linspace(0.45, 0.55, num=100)
geneSetA_size = len(geneSetA_ls)

for i in mu_As:
    sigma_A = 0.1
    new_geneA_dist = np.random.normal(i, sigma_A, geneSetA_size)
    sim_data.loc[test_sample,geneSetA_ls] = new_geneA_dist 
    A_exp_sample = sim_data.loc[test_sample]
    A_exp_sample_modified_df = A_exp_sample_modified_df.append(A_exp_sample, ignore_index=True)

A_exp_sample_modified_df.head()


# In[10]:


# Define function to apply latent space transformation to SHIFTED data and output reconstructed data

def interpolate_in_vae_latent_space_shiftA(all_data, 
                                       sample_data,
                                       model_encoder_file,
                                       model_decoder_file,
                                       weights_encoder_file,
                                       weights_decoder_file,
                                       encoded_dir,
                                       percent_low,
                                       percent_high,
                                       out_dir):
    """
    interpolate_in_vae_latent_space(all_data: dataframe,
                                    sample_data: dataframe,
                                    model_encoder_file: string,
                                    model_decoder_file: string,
                                    weights_encoder_file: string,
                                    weights_decoder_file: string,
                                    encoded_dir: string,
                                    gene_id: string,
                                    percent_low: integer,
                                    percent_high: integer,
                                    out_dir: string):

    input:
        all_data: Dataframe with gene expression data from all samples
        
        sample_data:  Dataframe with gene expression data from subset of samples (around the treshold)

        model_encoder_file: file containing the learned vae encoder model

        model_decoder_file: file containing the learned vae decoder model
        
        weights_encoder_file: file containing the learned weights associated with the vae encoder model
        
        weights_decoder_file: file containing the learned weights associated with the vae decoder model
        
        encoded_dir:  directory to use to output offset vector to 

        gene_id: gene you are using as the "phenotype" to sort samples by 

                 This gene is referred to as "target_gene" in comments below


        percent_low: integer between 0 and 1

        percent_high: integer between 0 and 1
        
        out_dir: directory to output predicted gene expression to

    computation:
        1.  Sort samples based on the expression level of the target gene defined by the user
        2.  Sample_data are encoded into VAE latent space
        3.  We predict the expression profile of the OTHER genes at a given level of target gene 
            expression by adding a scale factor of offset vector to the sample

            The scale factor depends on the distance along the target gene expression gradient
            the sample is.  For example the range along the target gene expression is from 0 to 1.  
            If the sample of interest has a target gene expression of 0.3 then our prediction
            for the gene expression of all other genes is equal to the gene expression corresponding
            to the target gene expression=0 + 0.3*offset latent vector
        3.  Prediction is decoded back into gene space
        4.  This computation is repeated for all samples 

    output: 
         1. encoded predicted expression profile per sample
         2. predicted expression profile per sample

    """

    # Load arguments
    offset_file = os.path.join(encoded_dir, "offset_latent_space_vae.txt")

    # Output file
    predict_file = os.path.join(out_dir, "shifted_predicted_gene_exp.txt")
    predict_encoded_file = os.path.join(out_dir, "shifted_predicted_encoded_gene_exp.txt")  
    
    # read in saved VAE models
    loaded_model = load_model(model_encoder_file)
    loaded_decoder_model = load_model(model_decoder_file)

    # load weights into models
    loaded_model.load_weights(weights_encoder_file)
    loaded_decoder_model.load_weights(weights_decoder_file)
    
    # Initialize dataframe for predicted expression of sampled data
    predicted_sample_data = pd.DataFrame(columns=sample_data.columns)
    predicted_encoded_sample_data = pd.DataFrame()
    
    sample_ids = sample_data.index
    for sample_id in sample_ids:
        sample_exp = sample_data.loc[sample_id].to_frame().T
        
        # Use trained model to encode expression data into SAME latent space
        predict = loaded_model.predict_on_batch(sample_exp)

        predict_encoded_df = pd.DataFrame(predict)
        
        predicted_encoded_sample_data = (
            predicted_encoded_sample_data
            .append(predict_encoded_df, ignore_index=True)
        )
        
        # Decode prediction
        predict_decoded = loaded_decoder_model.predict_on_batch(predict_encoded_df)
        predict_df = pd.DataFrame(
            predict_decoded, columns=sample_data.columns)
        
        predicted_sample_data = (
            predicted_sample_data
            .append(predict_df, ignore_index=True)
        )

    predicted_sample_data.set_index(sample_data.index, inplace=True)
    predicted_encoded_sample_data.set_index(sample_data.index, inplace=True)
    
    # Output estimated gene experession values
    predicted_sample_data.to_csv(predict_file, sep='\t')
    predicted_encoded_sample_data.to_csv(predict_encoded_file, sep='\t')


# In[11]:


# Apply function 
out_dir = os.path.join(base_dir, "output", analysis_name)
encoded_dir = os.path.join(base_dir, "encoded", analysis_name)

percent_low = 5
percent_high = 95
interpolate_in_vae_latent_space_shiftA(sim_data,
                                   A_exp_sample_modified_df,
                                   model_encoder_file,
                                   model_decoder_file,
                                   weights_encoder_file,
                                   weights_decoder_file,
                                   encoded_dir,
                                   percent_low,
                                   percent_high,
                                   out_dir)


# In[12]:


# Read dataframe with gene expression transformed
predict_file = os.path.join(base_dir, "output", analysis_name, "shifted_predicted_gene_exp.txt")
predict_gene_exp = pd.read_table(predict_file, header=0, index_col=0)

print(predict_gene_exp.shape)
predict_gene_exp.head()


# In[13]:


# Get the means of ORIGINAL B genes

# Convert dataframe with gene ids to list
geneSetB_ls = geneSetB['gene id'].values.tolist()

geneSetB_exp = A_exp_sample_modified_df[geneSetB_ls]

# Get the mean for each sample
geneSetB_mean_original = geneSetB_exp.mean(axis=1)
geneSetB_mean_original.head()


# In[14]:


# Get the means of ORIGINAL A genes

# Convert dataframe with gene ids to list
geneSetA_ls = geneSetA['gene id'].values.tolist()

geneSetA_exp = A_exp_sample_modified_df[geneSetA_ls]

# Get the mean for each sample
geneSetA_mean_original = geneSetA_exp.mean(axis=1)
geneSetA_mean_original.head()


# In[15]:


# Get the means of TRANSFORMED B genes

# Convert dataframe with gene ids to list
geneSetB_ls = geneSetB['gene id'].values.tolist()

geneSetB_exp = predict_gene_exp[geneSetB_ls]

# Get the mean for each sample
geneSetB_mean_transformed = geneSetB_exp.mean(axis=1)
geneSetB_mean_transformed.head()


# In[16]:


# Get the means of TRANSFORMED A genes

# Convert dataframe with gene ids to list
geneSetA_ls = geneSetA['gene id'].values.tolist()

geneSetA_exp = predict_gene_exp[geneSetA_ls]

# Get the mean for each sample
geneSetA_mean_transformed = geneSetA_exp.mean(axis=1)
geneSetA_mean_transformed.head()


# **Plot:** Original A vs Transformed A

# In[17]:


# Join original expression of A and transformed expression of A
original_A_mean_exp = geneSetA_mean_original
predict_A_mean_exp = geneSetA_mean_transformed

original_A_vs_transformed_A_df = pd.merge(original_A_mean_exp.to_frame('mean gene A untransformed'),
                                          predict_A_mean_exp.to_frame('mean gene A transformed'),
                                          left_index=True, right_index=True)

original_A_vs_transformed_A_df.head()


# In[18]:


# Plot
sns.regplot(x='mean gene A untransformed',
            y='mean gene A transformed',
           data = original_A_vs_transformed_A_df)


# **Plot:** Original A vs Mean(Transformed B)

# In[19]:


# Join original expression of A and mean(transformed expression of B)
original_A_mean_exp = geneSetA_mean_original
predict_B_mean_exp = geneSetB_mean_transformed

original_A_vs_transformed_B_df = pd.merge(original_A_mean_exp.to_frame('mean gene A untransformed'),
                      predict_B_mean_exp.to_frame('mean gene B transformed'),
                      left_index=True, right_index=True)

original_A_vs_transformed_B_df.head()


# In[20]:


# Plot
# A before transformation vs B after transformation
sns.regplot(x='mean gene A untransformed',
            y='mean gene B transformed',
           data = original_A_vs_transformed_B_df)


# **Plot:** Transformed A vs Mean(Transformed B)

# In[21]:


# Join original expression of transformed A and mean(transformed expression of B)
predict_A_mean_exp = geneSetA_mean_transformed
predict_B_mean_exp = geneSetB_mean_transformed

transformed_A_vs_transformed_B_df = pd.merge(predict_A_mean_exp.to_frame('mean gene A transformed'),
                      predict_B_mean_exp.to_frame('mean gene B transformed'),
                      left_index=True, right_index=True)
transformed_A_vs_transformed_B_df.head()


# In[22]:


# Plot
sns.regplot(x='mean gene A transformed',
            y='mean gene B transformed',
           data = transformed_A_vs_transformed_B_df)


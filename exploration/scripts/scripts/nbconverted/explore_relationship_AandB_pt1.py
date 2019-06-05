
# coding: utf-8

# # Explore simulated relationship
# 
# This notebook is using simulated data generated from [main_Pa_sim_enhance_AtoB](main_Pa_sim_enhance_AtoB.ipynb).  This notebook input raw Pseudomonas gene expression data from the Pseudomonas compendium referenced in [ADAGE](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5069748/) paper and added a strong nonlinear signal.  This signal assigned a set of genes to group A and a set of genes to group B.  If the expression of genes in group A exceeded some threshold then the genes in group B were upregulated.  
# 
# This notebook is tryign to determine if our VAE model is able to detect the relationship between A and B (i.e. when the expression of genes in set A exceed some threshold then the genes in set B are upregulated).
# 

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
analysis_name = 'sim_AB_2775_300_v2'

sim_data_file = os.path.join(base_dir, "data", analysis_name, "train_model_input.txt.xz")
A_file = os.path.join(base_dir, "data", analysis_name, "geneSetA.txt")
B_file = os.path.join(base_dir, "data", analysis_name, "geneSetB.txt")

offset_vae_file = os.path.join(os.path.dirname(os.getcwd()), "encoded", analysis_name, "offset_latent_space_vae.txt")

weight_file = os.path.join(os.path.dirname(os.getcwd()), "data", analysis_name, "VAE_weight_matrix.txt")

model_encoder_file = glob.glob(os.path.join(base_dir, "models", analysis_name, "*_encoder_model.h5"))[0]
weights_encoder_file = glob.glob(os.path.join(base_dir, "models", analysis_name, "*_encoder_weights.h5"))[0]
model_decoder_file = glob.glob(os.path.join(base_dir, "models", analysis_name, "*_decoder_model.h5"))[0]
weights_decoder_file = glob.glob(os.path.join(base_dir, "models", analysis_name, "*_decoder_weights.h5"))[0]


# In[4]:


# Read data
sim_data = pd.read_table(sim_data_file, index_col=0, header=0, compression='xz')
geneSetA = pd.read_table(A_file, header=0, index_col=0)
geneSetB = pd.read_table(B_file, header=0, index_col=0)

print(sim_data.shape)
sim_data.head()


# In[5]:


# Select samples that have expression of gene A around the threshold 
# Since threshold is 0.5 then select samples with expression in range(0.4, 0.6)

# Since our simulation set all genes in set A to be the same value for a give sample
# we can consider a single gene in set A to query by
rep_gene_A = geneSetA.iloc[0][0]

# Query for samples whose representative gene A expression is in range (0.4, 0.6)
#test_samples = sim_data.query('0.4 < @rep_gene_A < 0.6') -- why didn't this work?
test_samples = sim_data[(sim_data[rep_gene_A]>0.4) & (sim_data[rep_gene_A]<0.6)]

test_samples_sorted = test_samples.sort_values(by=[rep_gene_A])

print(test_samples_sorted.shape)
test_samples_sorted.head()


# ## Trend of gene B with respect to A (before transformation)
# 
# How is B changing with respect to A in our simulated dataset?
# 
# Plot gene expression of A vs mean(gene B expression).  This plot will serve as a reference against the later plot that will show gene expression of A vs mean(**transformed** gene B expression)

# In[6]:


# Get the means of B genes

# Convert dataframe with gene ids to list
geneSetB_ls = geneSetB['gene id'].values.tolist()

geneSetB_exp = test_samples[geneSetB_ls]

# Get the mean for each sample
geneSetB_mean = geneSetB_exp.mean(axis=1)
geneSetB_mean.head()


# In[7]:


# Join original expression of A and mean(transformed expression of B)
original_A_exp = test_samples[rep_gene_A]
original_B_mean_exp = geneSetB_mean

A_and_B_before_df = pd.merge(original_A_exp.to_frame('gene A untransformed'),
                      original_B_mean_exp.to_frame('mean gene B untransformed'),
                      left_index=True, right_index=True)
A_and_B_before_df.head()


# ### Plot
# The plot below shows the signal that was added to the dataset.  This signal assigned a set of genes to group A and a set of genes to group B. If the expression of genes in group A exceeded some threshold then the genes in group B were upregulated.  
# 
# So we see a step function relationship between the expression of genes in group A and the expression of genes in group B.  With a threshold of 0.5 we can see that the expression of genes in B are upregulated.

# In[8]:


# Plot
sns.regplot(x='gene A untransformed',
            y='mean gene B untransformed',
           data = A_and_B_before_df)


# ## Trend of gene B with respect to A (after transformation)
# 
# How is B changing with respect to A after applying our latent space transformation?

# In[9]:


# Define function to apply latent space transformation and output reconstructed data

def interpolate_in_vae_latent_space_AB(all_data, 
                                       sample_data,
                                       model_encoder_file,
                                       model_decoder_file,
                                       weights_encoder_file,
                                       weights_decoder_file,
                                       encoded_dir,
                                       gene_id,
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
    predict_file = os.path.join(out_dir, "predicted_gene_exp.txt")
    predict_encoded_file = os.path.join(out_dir, "predicted_encoded_gene_exp.txt")

    # Read in data
    target_gene_data = all_data[gene_id]
    offset_encoded = pd.read_table(offset_file, header=0, index_col=0)    
    
    # read in saved VAE models
    loaded_model = load_model(model_encoder_file)
    loaded_decoder_model = load_model(model_decoder_file)

    # load weights into models
    loaded_model.load_weights(weights_encoder_file)
    loaded_decoder_model.load_weights(weights_decoder_file)
    
    # Sort target gene data by expression (lowest --> highest)
    target_gene_sorted = target_gene_data.sort_values()

    lowest_file = os.path.join(encoded_dir, "lowest_encoded_vae.txt")
    low_exp_encoded = pd.read_table(lowest_file, header=0, index_col=0)
    
    # Average gene expression across samples in each extreme group
    lowest_mean_encoded = low_exp_encoded.mean(axis=0)

    # Format and rename as "baseline"
    baseline_encoded = pd.DataFrame(
        lowest_mean_encoded, index=offset_encoded.columns).T
    
    # Initialize dataframe for predicted expression of sampled data
    predicted_sample_data = pd.DataFrame(columns=sample_data.columns)
    predicted_encoded_sample_data = pd.DataFrame()
    
    sample_ids = sample_data.index
    for sample_id in sample_ids:
        intermediate_target_gene_exp = target_gene_sorted[sample_id]
        print('gene A exp is {}'.format(intermediate_target_gene_exp))
        alpha = get_scale_factor(
            target_gene_sorted, intermediate_target_gene_exp, percent_low, percent_high)
        print('scale factor is {}'.format(alpha))
        predict = baseline_encoded + alpha * offset_encoded

        predict_encoded_df = pd.DataFrame(predict)
        predicted_encoded_sample_data = predicted_encoded_sample_data.append(predict_encoded_df, ignore_index=True)
        
        # Decode prediction
        predict_decoded = loaded_decoder_model.predict_on_batch(predict)
        predict_df = pd.DataFrame(
            predict_decoded, columns=sample_data.columns)
        predicted_sample_data = predicted_sample_data.append(predict_df, ignore_index=True)

    predicted_sample_data.set_index(sample_data.index, inplace=True)
    predicted_encoded_sample_data.set_index(sample_data.index, inplace=True)
    
    # Output estimated gene experession values
    predicted_sample_data.to_csv(predict_file, sep='\t')
    predicted_encoded_sample_data.to_csv(predict_encoded_file, sep='\t')
    
def get_scale_factor(target_gene_sorted, expression_profile,
                     percent_low, percent_high):
    """
    get_scale_factor(target_gene_sorted: dataframe,
                    expression_profile: dataframe,
                    percent_low: integer,
                    percent_high: integer,):

    input:
        target_gene_sorted: dataframe of sorted target gene expression

        expression_profile: dataframe of gene expression for selected sample

        percent_low: integer between 0 and 1

        percent_high: integer between 0 and 1

    computation:
        Determine how much to scale offset based on distance along the target gene expression gradient

    Output:
     scale factor = intermediate gene expression/ (average high target gene expression - avgerage low target gene expression) 
    """

    # Collect the extreme gene expressions
    # Get sample IDs with the lowest 5% of reference gene expression
    threshold_low = np.percentile(target_gene_sorted, percent_low)
    lowest = target_gene_sorted[target_gene_sorted <= threshold_low]

    # Get sample IDs with the highest 5% of reference gene expression
    threshold_high = np.percentile(target_gene_sorted, percent_high)
    highest = target_gene_sorted[target_gene_sorted >= threshold_high]

    # Average gene expression across samples in each extreme group
    lowest_mean = (lowest.values).mean()
    highest_mean = (highest.values).mean()

    # Different in extremes
    denom = highest_mean - lowest_mean

    # scale_factor is the proportion along the gene expression gradient
    scale_factor = expression_profile / denom

    return scale_factor


# In[10]:


# Apply function 
out_dir = os.path.join(base_dir, "output", analysis_name)
encoded_dir = os.path.join(base_dir, "encoded", analysis_name)

percent_low = 5
percent_high = 95
interpolate_in_vae_latent_space_AB(sim_data,
                                   test_samples_sorted,
                                   model_encoder_file,
                                   model_decoder_file,
                                   weights_encoder_file,
                                   weights_decoder_file,
                                   encoded_dir,
                                   rep_gene_A,
                                   percent_low,
                                   percent_high,
                                   out_dir)


# ## Plot
# Plot gene expression A vs mean expression of genes in set B after transformation
# What is the relationship between genes in set A and B?  As the expression of A varies how does the expression of B vary?

# In[11]:


# Read dataframe with gene expression transformed
predict_file = os.path.join(base_dir, "output", analysis_name, "predicted_gene_exp.txt")
predict_gene_exp = pd.read_table(predict_file, header=0, index_col=0)

print(predict_gene_exp.shape)
predict_gene_exp.head()


# In[12]:


# Get the means of B genes

# Convert dataframe with gene ids to list
geneSetB_ls = geneSetB['gene id'].values.tolist()

geneSetB_exp = predict_gene_exp[geneSetB_ls]

# Get the mean for each sample
geneSetB_mean = geneSetB_exp.mean(axis=1)
geneSetB_mean.head()


# In[13]:


# Join original expression of A and mean(transformed expression of B)
original_A_exp = test_samples[rep_gene_A]
predict_B_mean_exp = geneSetB_mean

A_and_B_df = pd.merge(original_A_exp.to_frame('gene A untransformed'),
                      predict_B_mean_exp.to_frame('mean gene B transformed'),
                      left_index=True, right_index=True)
A_and_B_df.head()


# In[14]:


# Plot
# A before transformation vs B after transformation
sns.regplot(x='gene A untransformed',
            y='mean gene B transformed',
           data = A_and_B_df)


# In[40]:


# Join original expression of transformed A and mean(transformed expression of B)
predict_A_exp = predict_gene_exp[rep_gene_A]
predict_B_mean_exp = geneSetB_mean

A_and_B_predict_df = pd.merge(predict_A_exp.to_frame('gene A transformed'),
                      predict_B_mean_exp.to_frame('mean gene B transformed'),
                      left_index=True, right_index=True)
A_and_B_predict_df.head()


# In[41]:


# Plot
# A after transformation vs B after transformation
sns.regplot(x='gene A transformed',
            y='mean gene B transformed',
           data = A_and_B_predict_df)


# **Observations**:  This plot shows that the relationship that is learned by the VAE appears to be mostly linear but there is a slight kink around the threshold.  Recall that the input relationship we put into the dataset was a step function relationship.  
# 
# So now the question is why the VAE is learning a mostly linear relationship between A and B genes?  

# ## What is the offset capturing?
# 
# How are the values of the latent features changing as A moves across threshold? i.e. What is the difference in the encodings when gene A expression is above vs below the threshold?
# 
# This is the shift we should be capturing in our offset vector
# 
# Currently we are using the extremes of gene A to capture the “essence of A” and what it means to “turn on”

# In[16]:


# Read VAE space offset
offset_vae_space = pd.read_table(offset_vae_file, header=0, index_col=0)
offset_vae_space


# In[17]:


# Read dataframe with gene expression transformed
predict_encoded_file = os.path.join(base_dir, "output", analysis_name, "predicted_encoded_gene_exp.txt")
predict_encoded_gene_exp = pd.read_table(predict_encoded_file, header=0, index_col=0)

print(predict_encoded_gene_exp.shape)
predict_encoded_gene_exp.head()


# In[18]:


data_encoded = pd.concat([offset_vae_space, predict_encoded_gene_exp], axis=0)
data_encoded.head()


# ### Gene expression pattern around threshold
# 
# We looked at the samples around the threshold encoded in the latent space (300 dimensions).  
# 
# The heatmap shows that each row is a sample and the column is the latent space feature.  Each sample is grouped such that samples that are below the 0.5 threshold are labeled blue, at the threshold +/- 0.01 are green, above the 0.5 threshold are magenta.
# 
# We don't see a very clear trend by eye.

# In[19]:


# Add group labels per sample (<0.5, 0.5, 0.5>)
samples = data_encoded.index

data_encoded_labeled = data_encoded.assign(
    threshold_group=(
        list( 
            map(
                lambda x: 'offset' if x== 0 
                else 'less' if test_samples_sorted.loc[x,rep_gene_A]<0.49
                else 'threshold' if 0.49<= test_samples_sorted.loc[x,rep_gene_A]<=0.51
                else 'greater',
                samples
            )      
        )
    )
)
data_encoded_labeled.head()


# In[20]:


# Heatmap sorted by gene expression signature
# colormap: 
#     offset - red
#     less - blue
#     threshold - green
#     greater- magenta
threshold_groups = data_encoded_labeled["threshold_group"]
lut = dict(zip(threshold_groups.unique(), "rbgm"))
row_colors = threshold_groups.map(lut)

sns.clustermap(data_encoded,
               row_cluster=True,
               col_cluster=False,
               metric="correlation",
               row_colors=row_colors,
               figsize=(50,50),
              cmap='viridis')


# ### Collapse gene expression pattern
# It is difficult to see any trends in large heatmap so collapse the encoded gene expression using mean per group

# In[21]:


# Get mean of samples in each group ()

data_encoded_mean_less = pd.DataFrame(data_encoded_labeled[data_encoded_labeled.threshold_group == 'less'].mean(numeric_only=True))
data_encoded_mean_threshold = pd.DataFrame(data_encoded_labeled[data_encoded_labeled.threshold_group == 'threshold'].mean(numeric_only=True))
data_encoded_mean_greater = pd.DataFrame(data_encoded_labeled[data_encoded_labeled.threshold_group == 'greater'].mean(numeric_only=True))


# In[22]:


# Plot mean expression for each group
plt.figure(figsize=(100, 10))
sns.heatmap(data_encoded_mean_less.T, annot = True, cmap = "RdBu_r")
plt.figure(figsize=(100, 10))
sns.heatmap(data_encoded_mean_threshold.T, annot = True, cmap = "RdBu_r")
plt.figure(figsize=(100, 10))
sns.heatmap(data_encoded_mean_greater.T, annot = True, cmap = "RdBu_r")


# ### Which features have the largest difference as we cross the threshold?

# In[23]:


# Difference in means
diff_data_encoded = data_encoded_mean_greater - data_encoded_mean_less
abs_diff_data_encoded = abs(diff_data_encoded)

# Get top 5 features
top_features = abs_diff_data_encoded[0].nlargest()
top_features


# In[24]:


# Check sign of top features
feature_names = [int(l) for l in top_features.index]
diff_data_encoded.iloc[feature_names]


# In[25]:


# What is the weight of the offset vector for this top feature?
top_feature = top_features.index[0]
print(offset_vae_space[top_feature])

sns.distplot(offset_vae_space)


# ## What is the trend of latent features across the threshold?
# 
# Plot trend of encoded values along threshold for top feature

# In[26]:


# Join sorted original expression of A and encoded expression of top feature
original_A_exp = test_samples_sorted[rep_gene_A]
feature_values = data_encoded[top_feature]

trend_feature_df = pd.merge(original_A_exp.to_frame('gene A untransformed'),
                      feature_values.to_frame('values of feature {}'.format(top_feature)),
                      left_index=True, right_index=True)
trend_feature_df.head()


# In[27]:


# Plot
sns.regplot(x='gene A untransformed',
            y='values of feature {}'.format(top_feature),
           data = trend_feature_df)


# ## Which genes are highly weighted in the feature of significance?

# In[28]:


# Read in weight matrix
weight = pd.read_table(weight_file, header=0, index_col=0).T
weight.head(5)


# In[29]:


# Get genes associated with top feature
top_feature = int(top_features.index[0])
top_feature_genes = weight[top_feature]


# In[30]:


# Calculate mean per node ("signature" or "feature")
node_mean = top_feature_genes.mean()

# Calculate 2 standard deviations per node ("signature" or "feature")
stds = top_feature_genes.std()
two_stds = 2*stds

pos_threshold = node_mean + two_stds
neg_threshold = node_mean - two_stds
    
hw_pos_genes = top_feature_genes[top_feature_genes > pos_threshold].index
hw_neg_genes = top_feature_genes[top_feature_genes < neg_threshold].index

print(hw_pos_genes.shape)
hw_pos_genes


# In[31]:


print(hw_neg_genes.shape)
hw_neg_genes


# In[32]:


# Convert dataframe with gene ids to list
geneSetA_ls = geneSetA['gene id'].values.tolist()
geneSetB_ls = geneSetB['gene id'].values.tolist()

geneSetA_set = set(geneSetA_ls)
geneSetB_set = set(geneSetB_ls)


# In[33]:


# Compare the overlap of genes in set A and highest positive weighted genes in top feature
venn2([set(hw_pos_genes), geneSetA_set], set_labels = ('High weight pos genes', 'Group A genes'))
plt.show()


# In[34]:


# Compare the overlap of genes in set B and highest positive weighted genes in top feature
venn2([set(hw_pos_genes), geneSetB_set], set_labels = ('High weight pos genes', 'Group B genes'))
plt.show()


# In[35]:


# Compare the overlap of genes in set A and highest negative weighted genes in top feature
venn2([set(hw_neg_genes), geneSetA_set], set_labels = ('High weight neg genes', 'Group A genes'))
plt.show()


# In[36]:


# Compare the overlap of genes in set B and highest negative weighted genes in top feature
venn2([set(hw_neg_genes), geneSetB_set], set_labels = ('High weight neg genes', 'Group B genes'))
plt.show()


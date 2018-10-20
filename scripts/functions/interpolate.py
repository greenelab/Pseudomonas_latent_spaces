#-------------------------------------------------------------------------------------------------------------------------------
# By Alexandra Lee 
# (updated October 2018) 
# 
# Make predictions about intermediate gene expression using offset vector 
#-------------------------------------------------------------------------------------------------------------------------------
import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from keras.models import model_from_json, load_model
from keras import metrics, optimizers

from functions import utils

randomState = 123
from numpy.random import seed
seed(randomState)

def interpolate_gene_space(data_dir, gene_id, out_dir):
    """
    Interpolate in gene space:

    Add scale factor of offset vector to each sample to transform the gene expression 
     profile along some gradient
     
    Output: 
     Predicted expression profile per sample (intermediate samples x 2 statistical scores- correlation and pvalue)
     Target gene expression sorted by expression level for reference when plotting
    """
    
    # Load arguments
    target_gene_file = os.path.join(data_dir, "PA1673.txt")
    non_target_gene_file = os.path.join(data_dir, "train_model_input.txt.xz")
    offset_file = os.path.join(data_dir, "offset_gene_space.txt")

    # Output files
    corr_file = os.path.join(out_dir, "corr_gene_space.txt")
    sorted_file = os.path.join(out_dir, "sorted_id.txt")

    # Read in data
    target_gene_data = pd.read_table(target_gene_file, header=0, sep='\t', index_col=0)
    non_target_gene_data = pd.read_table(non_target_gene_file, header=0, sep='\t', index_col=0)
    offset_data = pd.read_table(offset_file, header=0, sep='\t', index_col=0)

    # Sort target gene data by expression (lowest --> highest)
    target_gene_sorted = target_gene_data.sort_values(by=[gene_id])

    # Get sample IDs with the lowest 5% of target gene expression
    threshold_low = np.percentile(target_gene_sorted[gene_id], 5)
    low_ids = target_gene_sorted[target_gene_sorted[gene_id]<= threshold_low].index
    low_exp = non_target_gene_data.loc[low_ids]

    # Average gene expression across samples in each extreme group
    lowest_mean = low_exp.mean(axis=0)

    # Format and rename as "baseline"
    baseline = pd.Series.to_frame(lowest_mean).T
  
    # Loop through all samples in the compendium in order of the target gene expression
    remaining_ids = target_gene_sorted[target_gene_sorted[gene_id]> threshold_low].index

    corr_score = {}
    for sample_id in remaining_ids:
        intermediate_target_gene_exp = target_gene_sorted.loc[sample_id]
        alpha = utils.get_scale_factor(target_gene_file, gene_id, intermediate_target_gene_exp)

        predict = baseline + alpha.values[0]*offset_data
        true = pd.Series.to_frame(non_target_gene_data.loc[sample_id]).T

        [coeff, pval] = pearsonr(predict.values.T, true.values.T)
        corr_score[sample_id] = coeff 

    corr_score_df = pd.DataFrame.from_dict(corr_score, orient='index')

    # Output estimated gene experession values
    corr_score_df.to_csv(corr_file, sep='\t')
    target_gene_sorted.to_csv(sorted_file, sep='\t')

def interpolate_latent_space(data_dir, model_dir, encoded_dir, gene_id, out_dir):
    """
    Interpolate in gene space:

    Add scale factor of offset vector to each encoded sample to transform the gene expression 
     profile along some gradient then decode
     
    Output: 
     Predicted expression profile per sample (intermediate samples x 2 statistical scores- correlation and pvalue)
    """
    
    # Load arguments
    target_gene_file = os.path.join(data_dir, "PA1673.txt")
    non_target_gene_file = os.path.join(data_dir, "train_model_input.txt.xz")
    offset_file = os.path.join(encoded_dir, "offset_latent_space.txt")

    model_file = os.path.join(model_dir, "tybalt_2layer_10latent_encoder_model.h5")
    weights_file = os.path.join(model_dir, "tybalt_2layer_10latent_encoder_weights.h5")
    model_decode_file = os.path.join(model_dir, "tybalt_2layer_10latent_decoder_model.h5")
    weights_decode_file = os.path.join(model_dir, "tybalt_2layer_10latent_decoder_weights.h5")

    # Output files
    corr_file = os.path.join(out_dir, "corr_latent_space.txt")

    # Read in data
    target_gene_data = pd.read_table(target_gene_file, header=0, sep='\t', index_col=0)
    non_target_gene_data = pd.read_table(non_target_gene_file, header=0, sep='\t', index_col=0)
    offset_encoded = pd.read_table(offset_file, header=0, sep='\t', index_col=0)

    # read in saved models
    loaded_model = load_model(model_file)
    loaded_decode_model = load_model(model_decode_file)

    # load weights into new model
    loaded_model.load_weights(weights_file)
    loaded_decode_model.load_weights(weights_decode_file)

    # Sort target gene data by expression (lowest --> highest)
    target_gene_sorted = target_gene_data.sort_values(by=[gene_id])

    # Get sample IDs with the lowest 5% of target gene expression
    threshold_low = np.percentile(target_gene_sorted[gene_id], 5)
    low_ids = target_gene_sorted[target_gene_sorted[gene_id]<= threshold_low].index
    low_exp = non_target_gene_data.loc[low_ids]

    # Use trained model to encode expression data into SAME latent space
    low_exp_encoded = loaded_model.predict_on_batch(low_exp)
    low_exp_encoded_df = pd.DataFrame(low_exp_encoded, index=low_exp.index)

    # Average gene expression across samples in each extreme group
    lowest_mean_encoded = low_exp_encoded.mean(axis=0)

    # Format and rename as "baseline"
    baseline_encoded = pd.DataFrame(lowest_mean_encoded, index=offset_encoded.columns).T
    baseline_encoded

    # Loop through all samples in the compendium in order of the target gene expression
    remaining_ids = target_gene_sorted[target_gene_sorted[gene_id]> threshold_low].index

    corr_score = {}
    for sample_id in remaining_ids:
        intermediate_target_gene_exp = target_gene_sorted.loc[sample_id]
        alpha = utils.get_scale_factor(target_gene_file, gene_id, intermediate_target_gene_exp)

        predict = baseline_encoded + alpha.values[0]*offset_encoded

        # Decode prediction
        predict_decoded = loaded_decode_model.predict_on_batch(predict)
        predict = pd.DataFrame(predict_decoded, columns=non_target_gene_data.columns)

        true = pd.Series.to_frame(non_target_gene_data.loc[sample_id]).T

        [coeff, pval] = pearsonr(predict.values.T, true.values.T)
        corr_score[sample_id] = coeff 

    corr_score_df = pd.DataFrame.from_dict(corr_score, orient='index')

    # Output estimated gene experession values
    corr_score_df.to_csv(corr_file, sep='\t')
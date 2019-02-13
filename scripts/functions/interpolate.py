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
import pickle

from functions import utils

randomState = 123
from numpy.random import seed
seed(randomState)


def interpolate_in_gene_space(data_dir, gene_id, out_dir,
                              percent_low, percent_high):
    """
    interpolate_in_gene_space(data_dir: string, gene_id: string, out_dir: string):

    input:
        data_dir: directory containing the raw gene expression data and the offset vector

        gene_id: gene you are using as the "phenotype" to sort samples by 

                 This gene is referred to as "target_gene" in comments below

        out_dir: directory to output predicted gene expression to

        percent_low: integer between 0 and 1

        percent_high: integer between 0 and 1

    computation:
        1.  Sort samples based on the expression level of the target gene defined by the user
        2.  We predict the expression profile of the OTHER genes at a given level of target gene 
            expression by adding a scale factor of offset vector to the sample

            The scale factor depends on the distance along the target gene expression gradient
            the sample is.  For example the range along the target gene expression is from 0 to 1.  
            If the sample of interest has a target gene expression of 0.3 then our prediction
            for the gene expression of all other genes is equal to gene expression corresponding
            to the target gene expression=0 + 0.3*offset vector
        3.  This computation is repeated for all samples 

    output: 
         1. predicted expression profile per sample (intermediate samples x 2 statistical scores --> correlation and pvalue)
         2. target gene expression sorted by expression level for reference when plotting

    """

    # Load arguments
    target_gene_file = os.path.join(data_dir, gene_id + ".txt")
    non_target_gene_file = os.path.join(data_dir, "train_model_input.txt.xz")
    offset_file = os.path.join(data_dir, "offset_gene_space.txt")

    # Output files
    corr_file = os.path.join(out_dir, "corr_gene_space.txt")
    sorted_file = os.path.join(out_dir, "sorted_id.txt")

    # Read in data
    target_gene_data = pd.read_table(target_gene_file, header=0, index_col=0)
    non_target_gene_data = pd.read_table(
        non_target_gene_file, header=0, index_col=0)
    offset_data = pd.read_table(offset_file, header=0, index_col=0)

    # Sort target gene data by expression (lowest --> highest)
    target_gene_sorted = target_gene_data.sort_values(by=[gene_id])

    # Get sample IDs with the lowest 5% of target gene expression
    [low_ids, high_ids] = utils.get_gene_expression_above_percent(
        target_gene_sorted, gene_id, percent_low, percent_high)
    low_exp = non_target_gene_data.loc[low_ids]

    # Average gene expression across samples in each extreme group
    lowest_mean = low_exp.mean(axis=0)

    # Format and rename as "baseline"
    baseline = pd.Series.to_frame(lowest_mean).T

    # Loop through all samples in the compendium in order of the target gene expression
    threshold_low = np.percentile(target_gene_sorted[gene_id], percent_low)
    remaining_ids = target_gene_sorted[target_gene_sorted[gene_id]
                                       > threshold_low].index

    corr_score = {}
    for sample_id in remaining_ids:
        intermediate_target_gene_exp = target_gene_sorted.loc[sample_id]
        alpha = utils.get_scale_factor(
            target_gene_sorted, gene_id, intermediate_target_gene_exp, percent_low, percent_high)
        
        predict = baseline + alpha.values[0] * offset_data
        true = pd.Series.to_frame(non_target_gene_data.loc[sample_id]).T
    
        [coeff, pval] = pearsonr(predict.values.T, true.values.T)
        corr_score[sample_id] = coeff

    corr_score_df = pd.DataFrame.from_dict(corr_score, orient='index')

    # Output estimated gene experession values
    corr_score_df.to_csv(corr_file, sep='\t')
    target_gene_sorted.to_csv(sorted_file, sep='\t', float_format="%.5g")


def interpolate_in_vae_latent_space(data_dir, model_dir, encoded_dir, latent_dim, 
                                    gene_id, out_dir,
                                    percent_low, percent_high):
    """
    interpolate_in_vae_latent_space(data_dir: string, gene_id: string, out_dir: string):

    input:
        data_dir: directory containing the raw gene expression data and the offset vector

        model_dir: directory containing the learned vae models

        encoded_dir: directory to use to output offset vector to 

        gene_id: gene you are using as the "phenotype" to sort samples by 

                 This gene is referred to as "target_gene" in comments below

        out_dir: directory to output predicted gene expression to

        percent_low: integer between 0 and 1

        percent_high: integer between 0 and 1

    computation:
        1.  Sort samples based on the expression level of the target gene defined by the user
        2.  Samples are encoded into VAE latent space
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
         1. predicted expression profile per sample (intermediate samples x 2 statistical scores --> correlation and pvalue)
         2. target gene expression sorted by expression level for reference when plotting

    """

    # Load arguments
    target_gene_file = os.path.join(data_dir, gene_id + ".txt")
    non_target_gene_file = os.path.join(data_dir, "train_model_input.txt.xz")
    offset_file = os.path.join(encoded_dir, "offset_latent_space_vae.txt")

    model_file = os.path.join(
        model_dir, "tybalt_2layer_{}latent_encoder_model.h5".format(latent_dim))
    weights_file = os.path.join(
        model_dir, "tybalt_2layer_{}latent_encoder_weights.h5".format(latent_dim))
    model_decode_file = os.path.join(
        model_dir, "tybalt_2layer_{}latent_decoder_model.h5".format(latent_dim))
    weights_decode_file = os.path.join(
        model_dir, "tybalt_2layer_{}latent_decoder_weights.h5".format(latent_dim))

    # Output files
    corr_file = os.path.join(out_dir, "corr_latent_space_vae.txt")

    # Read in data
    target_gene_data = pd.read_table(target_gene_file, header=0, index_col=0)
    non_target_gene_data = pd.read_table(
        non_target_gene_file, header=0, index_col=0)
    offset_encoded = pd.read_table(offset_file, header=0, index_col=0)

    # read in saved models
    loaded_model = load_model(model_file)
    loaded_decode_model = load_model(model_decode_file)

    # load weights into new model
    loaded_model.load_weights(weights_file)
    loaded_decode_model.load_weights(weights_decode_file)

    # Sort target gene data by expression (lowest --> highest)
    target_gene_sorted = target_gene_data.sort_values(by=[gene_id])

    # Get sample IDs with the lowest 5% of target gene expression
    [low_ids, high_ids] = utils.get_gene_expression_above_percent(
        target_gene_sorted, gene_id, 5, 95)
    low_exp = non_target_gene_data.loc[low_ids]

    # Use trained model to encode expression data into SAME latent space
    low_exp_encoded = loaded_model.predict_on_batch(low_exp)
    low_exp_encoded_df = pd.DataFrame(low_exp_encoded, index=low_exp.index)

    # Average gene expression across samples in each extreme group
    lowest_mean_encoded = low_exp_encoded.mean(axis=0)

    # Format and rename as "baseline"
    baseline_encoded = pd.DataFrame(
        lowest_mean_encoded, index=offset_encoded.columns).T

    # Loop through all samples in the compendium in order of the target gene expression
    threshold_low = np.percentile(target_gene_sorted[gene_id], percent_low)
    remaining_ids = target_gene_sorted[target_gene_sorted[gene_id]
                                       > threshold_low].index

    corr_score = {}
    for sample_id in remaining_ids:
        intermediate_target_gene_exp = target_gene_sorted.loc[sample_id]
        alpha = utils.get_scale_factor(
            target_gene_sorted, gene_id, intermediate_target_gene_exp, percent_low, percent_high)

        predict = baseline_encoded + alpha.values[0] * offset_encoded

        # Decode prediction
        predict_decoded = loaded_decode_model.predict_on_batch(predict)
        predict = pd.DataFrame(
            predict_decoded, columns=non_target_gene_data.columns)

        #print("predict is {}".format(predict))
        true = pd.Series.to_frame(non_target_gene_data.loc[sample_id]).T

        #print("true is {}".format(true))
        [coeff, pval] = pearsonr(predict.values.T, true.values.T)
        corr_score[sample_id] = coeff
    
        #break
    corr_score_df = pd.DataFrame.from_dict(corr_score, orient='index')

    # Output estimated gene experession values
    corr_score_df.to_csv(corr_file, sep='\t')


def interpolate_in_pca_latent_space(data_dir, model_dir, encoded_dir, gene_id,
                                    out_dir, percent_low, percent_high):
    """
    interpolate_in_pca_latent_space(data_dir: string, gene_id: string, out_dir: string):

    input:
        data_dir: directory containing the raw gene expression data and the offset vector

        model_dir: directory containing the learned vae models

        encoded_dir: directory to use to output offset vector to 

        gene_id: gene you are using as the "phenotype" to sort samples by 

                 This gene is referred to as "target_gene" in comments below

        out_dir: directory to output predicted gene expression to

        percent_low: integer between 0 and 1

        percent_high: integer between 0 and 1

    computation:
        1.  Sort samples based on the expression level of the target gene defined by the user
        2.  Samples are encoded into PCA latent space
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
         1. predicted expression profile per sample (intermediate samples x 2 statistical scores --> correlation and pvalue)
         2. target gene expression sorted by expression level for reference when plotting

    """
    # Load arguments
    target_gene_file = os.path.join(data_dir, gene_id + ".txt")
    non_target_gene_file = os.path.join(data_dir, "train_model_input.txt.xz")
    offset_file = os.path.join(encoded_dir, "offset_latent_space_pca.txt")

    model_file = os.path.join(model_dir, "pca_model.pkl")

    # Output files
    corr_file = os.path.join(out_dir, "corr_latent_space_pca.txt")

    # Read in data
    target_gene_data = pd.read_table(target_gene_file, header=0, index_col=0)
    non_target_gene_data = pd.read_table(
        non_target_gene_file, header=0, index_col=0)
    offset_encoded = pd.read_table(offset_file, header=0, index_col=0)

    # load pca model
    infile = open(model_file, 'rb')
    pca = pickle.load(infile)
    infile.close()

    # Sort target gene data by expression (lowest --> highest)
    target_gene_sorted = target_gene_data.sort_values(by=[gene_id])

    # Get sample IDs with the lowest 5% of target gene expression
    [low_ids, high_ids] = utils.get_gene_expression_above_percent(
        target_gene_sorted, gene_id, 5, 95)
    low_exp = non_target_gene_data.loc[low_ids]

    # Use trained model to encode expression data into SAME latent space
    low_exp_encoded = pca.transform(low_exp)
    low_exp_encoded_df = pd.DataFrame(low_exp_encoded, index=low_exp.index)

    # Average gene expression across samples in each extreme group
    lowest_mean_encoded = low_exp_encoded.mean(axis=0)

    # Format and rename as "baseline"
    baseline_encoded = pd.DataFrame(
        lowest_mean_encoded, index=offset_encoded.columns).T

    # Loop through all samples in the compendium in order of the target gene expression
    threshold_low = np.percentile(target_gene_sorted[gene_id], percent_low)
    remaining_ids = target_gene_sorted[target_gene_sorted[gene_id]
                                       > threshold_low].index

    corr_score = {}
    for sample_id in remaining_ids:
        intermediate_target_gene_exp = target_gene_sorted.loc[sample_id]
        alpha = utils.get_scale_factor(
            target_gene_sorted, gene_id, intermediate_target_gene_exp, percent_low, percent_high)

        predict = baseline_encoded + alpha.values[0] * offset_encoded

        # Decode prediction
        predict_decoded = pca.inverse_transform(predict)
        predict = pd.DataFrame(
            predict_decoded, columns=non_target_gene_data.columns)

        true = pd.Series.to_frame(non_target_gene_data.loc[sample_id]).T

        [coeff, pval] = pearsonr(predict.values.T, true.values.T)
        corr_score[sample_id] = coeff

    corr_score_df = pd.DataFrame.from_dict(corr_score, orient='index')

    # Output estimated gene experession values
    corr_score_df.to_csv(corr_file, sep='\t')

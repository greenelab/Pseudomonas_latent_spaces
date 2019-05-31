#-------------------------------------------------------------------------------------------------------------------------------
# By Alexandra Lee
# (updated May 2019)
#
# Encode - decode input data using trained compression model
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


def vae_encode_decode_shiftA(sample_data,
                             model_encoder_file,
                             model_decoder_file,
                             weights_encoder_file,
                             weights_decoder_file,
                             encoded_dir,
                             gene_id,
                             out_dir,
                             seed_input):
    """
    vae_encode_decode_shiftA(sample_data: dataframe,
                             model_encoder_file: string,
                             model_decoder_file: string,
                             weights_encoder_file: string,
                             weights_decoder_file: string,
                             encoded_dir: string,
                             gene_id: string,
                             out_dir: string,
                             seed_input: int):

    input:

        sample_data:  Dataframe with gene expression data from subset of samples (around the treshold)

        model_encoder_file: file containing the learned vae encoder model

        model_decoder_file: file containing the learned vae decoder model

        weights_encoder_file: file containing the learned weights associated with the vae encoder model

        weights_decoder_file: file containing the learned weights associated with the vae decoder model

        encoded_dir:  directory to use to output offset vector to 

        gene_id: gene you are using as the "phenotype" to sort samples by 

                 This gene is referred to as "target_gene" in comments below

        out_dir: directory to output predicted gene expression to

        seed_input: random seed

    computation:
        1.  Sort samples based on the expression level of the target gene defined by the user
        2.  Sample_data are encoded into VAE latent space
        3.  Encoded expression is decoded back into gene space
        4.  This computation is repeated for all samples 

    output: 
         1. encoded expression profile per sample
         2. predicted expression profile per sample

    """

    # Output file
    predict_file = os.path.join(out_dir,
                                "vae_predicted_gene_exp_seed" + str(seed_input) + ".txt")
    predict_encoded_file = os.path.join(
        out_dir, "vae_predicted_encoded_gene_exp_seed" + str(seed_input) + ".txt")

    # Read in saved VAE models
    loaded_model = load_model(model_encoder_file)
    loaded_decoder_model = load_model(model_decoder_file)

    # Load weights into models
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
        predict_decoded = loaded_decoder_model.predict_on_batch(
            predict_encoded_df)
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


def pca_encode_decode_shiftA(sample_data,
                             model_dir,
                             encoded_dir,
                             gene_id,
                             out_dir,
                             seed_input):
    """
    interpolate_in_pca_latent_space(sample_data: dataframe,
                                    model_dir: string,
                                    encoded_dir: string,
                                    gene_id: string,
                                    out_dir: string,
                                                                        seed_input: int):

    input:
        sample_data:  Dataframe with gene expression data from subset of samples (around the treshold)

        model_dir: directory containing the learned vae models

        encoded_dir: directory to use to output offset vector to 

        gene_id: gene you are using as the "phenotype" to sort samples by 

                 This gene is referred to as "target_gene" in comments below

        out_dir: directory to output predicted gene expression to

        seed_input: random seed


    computation:
        1.  Sort samples based on the expression level of the target gene defined by the user
        2.  Samples are encoded into PCA latent space
        3.  Prediction is decoded back into gene space
        4.  This computation is repeated for all samples 

    output: 
         1. encoded expression profile per sample
         2. predicted expression profile per sample
    """

	# Output file
    predict_file = os.path.join(out_dir,
                                "pca_predicted_gene_exp_seed" + str(seed_input) + ".txt")
    predict_encoded_file = os.path.join(
        out_dir, "pca_predicted_encoded_gene_exp_seed" + str(seed_input) + ".txt")

    # Load arguments
    model_file = os.path.join(
        model_dir, "pca_model_seed" + str(seed_input) + ".pkl")

    # Load pca model
    infile = open(model_file, 'rb')
    pca = pickle.load(infile)
    infile.close()

    # Initialize dataframe for predicted expression of sampled data
    predicted_sample_data = pd.DataFrame(columns=sample_data.columns)
    predicted_encoded_sample_data = pd.DataFrame()

    sample_ids = sample_data.index
    for sample_id in sample_ids:
        sample_exp = sample_data.loc[sample_id].to_frame().T

        # Use trained model to encode expression data into SAME latent space
        predict = pca.transform(sample_exp)

        predict_encoded_df = pd.DataFrame(predict)

        predicted_encoded_sample_data = (
            predicted_encoded_sample_data
            .append(predict_encoded_df, ignore_index=True)
        )

        # Decode prediction
        predict_decoded = pca.inverse_transform(predict_encoded_df)
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

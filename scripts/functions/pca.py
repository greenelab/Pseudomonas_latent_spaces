#-------------------------------------------------------------------------------------------------------------------------------
# By Alexandra Lee (July 2018)
#
# Apply PCA to compress Pseudomonas gene expression data
#
# Input: Pa gene expression data from ArrayExpress (matrix: sample x gene)
# Data compression method: PCA
# Output: Reduced Pa gene expressiond ata (matrix: sample x 2 linear combination of genes)
#-------------------------------------------------------------------------------------------------------------------------------
import os
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import pickle
from numpy.random import seed


def pca_model(base_dir,
              analysis_name,
              num_PCs):
    """
    Uses PCA to compress input data

    Return saved PCA model
    """

    seed(123)

    data_file = os.path.join(
        base_dir, "data", analysis_name, "train_model_input.txt.xz")
    rnaseq = pd.read_table(data_file, index_col=0, header=0, compression='xz')

    # Make an instance of the model
    pca = PCA(n_components=num_PCs)

    # Fit PCA on training dataset
    pca_model = pca.fit(rnaseq)

    # Output PCA model
    # save the model to disk
    file_out = os.path.join(base_dir, "models", analysis_name, "pca_model.pkl")
    pickle.dump(pca_model, open(file_out, 'wb'))


def pca_model_multi(base_dir,
                    analysis_name,
                    num_PCs,
                    seed_input):
    """
    Uses PCA to compress input data

    Return saved PCA model
    """

    seed(seed_input)

    data_file = os.path.join(
        base_dir, "data", analysis_name, "train_model_input_seed" + str(seed_input) + ".txt.xz")
    rnaseq = pd.read_table(data_file, index_col=0, header=0, compression='xz')

    # Make an instance of the model
    pca = PCA(n_components=num_PCs)

    # Fit PCA on training dataset
    pca_model = pca.fit(rnaseq)

    # Output PCA model
    # save the model to disk
    file_out = os.path.join(base_dir, "models", analysis_name,
                            "pca_model_seed" + str(seed_input) + ".pkl")
    pickle.dump(pca_model, open(file_out, 'wb'))

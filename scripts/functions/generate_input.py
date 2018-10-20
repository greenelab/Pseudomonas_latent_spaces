#-------------------------------------------------------------------------------------------------------------------------------
# By Alexandra Lee 
# (updated October 2018) 
#
# Generate input files for each analysis
#-------------------------------------------------------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np

randomState = 123
from numpy.random import seed
seed(randomState)

def generate_input_PA1673_gradient(base_dir):
    """
    Dataset: Pseudomonas aeruginosa gene expression from compendium 
    referenced in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5069748/

    Condition: expression of PA1673 gene

    Processing:
     1. Remove PA1673 gene that will be used as the "condition" (phenotype) for this analysis
     
    Output:  
     Dataframe (1191 samples x 5548 genes) 
    """
    
    # Load arguments
    data_file = os.path.join(os.path.dirname(os.getcwd()), "data", "all-pseudomonas-gene-normalized.zip")
    gene_id = "PA1673"

    # Output
    train_input_file = os.path.join(base_dir, "train_model_input.txt.xz")
    PA1673_file = os.path.join(base_dir, "PA1673.txt")


    # Read in data
    data = pd.read_table(data_file, header=0, sep='\t', index_col=0, compression='zip')
    X = data.transpose()
    X.head(5)


    # Dataframe with only PA1673 gene
    PA1673_only = pd.DataFrame(X[gene_id], index=X.index, columns=[gene_id])
    PA1673_only


    # Drop PA1673 gene
    input_holdout = X.drop(columns=[gene_id])
    input_holdout.head(5)


    # Output
    input_holdout.to_csv(train_input_file, sep='\t', compression='xz')
    PA1673_only.to_csv(PA1673_file, sep='\t')


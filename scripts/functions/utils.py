#----------------------------------------------------------------------------------------------------------------
# Helper functions
#----------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import os


def get_scale_factor(target_gene_sorted, gene_id, expression_profile,
                     percent_low, percent_high):
    """
    get_scale_factor(target_gene_sorted: df, gene_id: string, expression_profile: df):

    input:
        target_gene_sorted: dataframe of sorted target gene expression

        gene_id: gene you are using as the "phenotype" to sort samples by

        percent_low: integer between 0 and 1

        percent_high: integer between 0 and 1

    computation:
        Determine how much to scale offset based on distance along the target gene expression gradient

    Output:
     scale factor = intermediate gene expression/ (average high target gene expression - avgerage low target gene expression) 
    """

    # Collect the extreme gene expressions
    # Get sample IDs with the lowest 5% of reference gene expression
    threshold_low = np.percentile(target_gene_sorted[gene_id], percent_low)
    lowest = target_gene_sorted[target_gene_sorted[gene_id] <= threshold_low]

    # Get sample IDs with the highest 5% of reference gene expression
    threshold_high = np.percentile(target_gene_sorted[gene_id], percent_high)
    highest = target_gene_sorted[target_gene_sorted[gene_id] >= threshold_high]

    # Average gene expression across samples in each extreme group
    lowest_mean = (lowest.values).mean()
    highest_mean = (highest.values).mean()

    # Different in extremes
    denom = highest_mean - lowest_mean

    # scale_factor is the proportion along the gene expression gradient
    scale_factor = expression_profile / denom

    return scale_factor


def get_gene_expression_above_percent(target_gene_sorted, gene_id,
                                      percent_low, percent_high):
    """
    get_gene_expression_above_threshold(target_gene_sorted: df, gene_id: string, percent_low: int, percent_high: int):

    input:
        target_sorted: python object of target gene expression sorted

    gene_id: gene you are using as the "phenotype" to sort samples by

    percent_low: integer between 0 and 1

    percent_high: integer between 0 and 1

    output:
        return sample IDs with lowest [percent_low] target gene expression
        return sample IDs with highest [percent_high] target gene expression   
    """

    # Get sample IDs with the lowest [percent_low] of target gene expression
    threshold_low = np.percentile(target_gene_sorted[gene_id], percent_low)
    low_ids = target_gene_sorted[target_gene_sorted[gene_id]
                                 <= threshold_low].index

    # Get sample IDs with the highest [percent_high] of target gene expression
    threshold_high = np.percentile(target_gene_sorted[gene_id], percent_high)
    high_ids = target_gene_sorted[target_gene_sorted[gene_id]
                                  >= threshold_high].index

    return [low_ids, high_ids]

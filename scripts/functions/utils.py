#----------------------------------------------------------------------------------------------------------------
# Helper functions
#----------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import os

def get_scale_factor (ref_file, gene_id, expression_profile):
    

    """
    Determine how much to scale offset based on distance along the target gene expression gradient
    
    Output:
     scale factor = intermediate gene expression/ (average high target gene expression - avgerage low target gene expression) 
    """

    # Read in reference gene expression file
    ref_data = pd.read_table(ref_file, header=0, sep='\t', index_col=0)

    # Sort reference gene expression (lowest --> highest)
    ref_sorted = ref_data.sort_values(by=[gene_id])

    # Collect the extreme gene expressions
    # Get sample IDs with the lowest 5% of reference gene expression
    threshold_low = np.percentile(ref_sorted[gene_id], 5)
    lowest = ref_sorted[ref_sorted[gene_id]<= threshold_low]

    # Get sample IDs with the highest 5% of reference gene expression
    threshold_high = np.percentile(ref_sorted[gene_id], 95)
    highest = ref_sorted[ref_sorted[gene_id]>= threshold_high]

    #print(lowest)
    #print(highest)

    # Average gene expression across samples in each extreme group
    lowest_mean = (lowest.values).mean()
    highest_mean = (highest.values).mean()

    # Different in extremes
    denom = highest_mean - lowest_mean

    # scale_factor is the proportion along the gene expression gradient
    scale_factor = expression_profile/denom
    
    return scale_factor


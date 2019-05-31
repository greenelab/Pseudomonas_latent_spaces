import os
import pandas as pd
import numpy as np
import random
from numpy.random import seed


def simulate_data(
        geneSetA_size,
        geneSetB_size,
        effect_size,
        thresholdA,
        analysis_name,
        seed_input):

    seed(seed_input)

    # Create list of base directories
    base_dirs = [os.path.join(os.path.dirname(os.getcwd()), 'data'),
                 os.path.join(os.path.dirname(os.getcwd()), 'encoded'),
                 os.path.join(os.path.dirname(os.getcwd()), 'models'),
                 os.path.join(os.path.dirname(os.getcwd()), 'output'),
                 os.path.join(os.path.dirname(os.getcwd()), 'stats'),
                 os.path.join(os.path.dirname(os.getcwd()), 'viz')
                 ]

    # Check if directory exist otherwise create
    for each_dir in base_dirs:
        analysis_dir = os.path.join(each_dir, analysis_name)

        if os.path.exists(analysis_dir):
            print('directory already exists: {}'.format(analysis_dir))
        else:
            print('creating new directory: {}'.format(analysis_dir))
        os.makedirs(analysis_dir, exist_ok=True)

        # Load arguments
        data_file = os.path.join(os.path.dirname(os.getcwd()),
                                 "data", "train_set_normalized.pcl")

    # Read in data
    data = pd.read_table(data_file, header=0, sep='\t',
                         index_col=0).T

    # Randomly select genes for set A
    gene_ids = list(data.columns)
    geneSetA = random.sample(gene_ids, geneSetA_size)

    # checkpoint:  Check the number of genes
    assert(len(gene_ids) == 5549)
    print("Confirmed that all gene ids are available")

    # Remove gene ids that were selected for gene set A
    gene_ids = [elem for elem in gene_ids if elem not in geneSetA]
    print("After removing {} gene ids for set A, there are {} gene ids remaining."
          .format(geneSetA_size, len(gene_ids)))

    # Randomly select genes for gene set B
    geneSetB = random.sample(gene_ids, geneSetB_size)

    # Remove gene ids that were selected for gene set B
    gene_ids = [elem for elem in gene_ids if elem not in geneSetB]
    print("After removing {} gene ids for set B, there are {} gene ids remaining."
          .format(geneSetB_size, len(gene_ids)))

    # checkpoint:  Check that genes in set A are distinct from genes in set B
    assert(len(set(geneSetA).intersection(set(geneSetB))) == 0)

    # Output gene groupings
    # Output gene assignments (group A, B) to be used in [explore_simulated_data.py](explore_simulated_data.ipynb)

    geneSetA_df = pd.DataFrame(geneSetA, columns=['gene id'])
    geneSetB_df = pd.DataFrame(geneSetB, columns=['gene id'])

    geneSetA_file = os.path.join(os.path.dirname(
        os.getcwd()), "data", analysis_name, "geneSetA_seed" + str(seed_input) + ".txt")
    geneSetB_file = os.path.join(os.path.dirname(
        os.getcwd()), "data", analysis_name, "geneSetB_seed" + str(seed_input) + ".txt")

    geneSetA_df.to_csv(geneSetA_file, sep='\t')
    geneSetB_df.to_csv(geneSetB_file, sep='\t')

    # Add artificial signal to the data
    # Algorithm:
    # ```python
    # for sample in compendium:
    #     expression(gene_in_set_A) = random(0,1)
    #     if expression(gene_in_set_A) > threshold_A:
    #         expression(gene_in_set_B) = expression(gene_in_set_B)*(1+percentage)
    # ```
    # Note: This algorithm is applied to each sample in the compendium
    #       (essentially adding a signal to the existing gene expression data in the compendium)

    # Number of samples
    num_samples = data.shape[1]

    # Select sample gene from set A to be representative since the expression is the same
    # for all genes in the set
    geneSetA_pick = geneSetA[0]

    # Loop through all samples
    for sample_id in data.index:

        # Randomly select a value [0,1] where each value is uniformly likely to be chosen
        new_A_exp = random.uniform(0.0, 1.0)

        # Set gene expression value for genes in set A to be the same random value selected
        data.loc[sample_id][geneSetA] = new_A_exp

        # Check if expression of genes in set A exceed the threshold
        # Use representatve gene from set A "geneSetA_pick" selected above
        if data.loc[sample_id, geneSetA_pick] > thresholdA:

            # Scale genes by some fixed percentage
            for gene in geneSetB:
                data.loc[sample_id, gene] = (
                    1 + effect_size) * data.loc[sample_id, gene]

    # if any exceed 1 then set to 1 since gene expression is normalized
    data[data >= 1.0] = 1.0

    # Dataframe with only gene A expression
    # Used in interpolation analysis
    geneA_only = pd.DataFrame(
        data[geneSetA_pick], index=data.index, columns=[geneSetA_pick])

    geneA_only.head()

    # Output
    # 1. Simulated dataset (\data)
    # 2. Log file containing hyperparmeters used (\metadata)
    # 3. Expression of representative gene from group A (\data)

    # Output the new gene expression values for each sample
    train_input_file = os.path.join(
        base_dirs[0], analysis_name, "train_model_input_seed" + str(seed_input) + ".txt.xz")

    # Only include genes in group A and B
    geneSetAB = geneSetA + geneSetB
    simplified_data = data[geneSetAB]
    simplified_data.to_csv(train_input_file, sep='\t',
                           compression='xz', float_format="%.5g")

    # Output log file with parameters used to generate simulated data
    log_file = os.path.join(os.path.dirname(os.getcwd()),
                            "metadata", analysis_name + "_log.txt")

    args_dict = {
        "size of gene set A": geneSetA_size,
        "size of gene set B": geneSetB_size,
        "genes in set B": geneSetB,
        "effect size of genes in set B": effect_size,
        "threshold of A activation": thresholdA,
    }

    with open(log_file, 'w') as f:
        for key, value in args_dict.items():
            f.write('%s: %s\n' % (key, value))

    # Output geneA only file using sample gene A selected
    geneA_file = os.path.join(
        base_dirs[0], analysis_name, geneSetA_pick + ".txt")
    geneA_only.to_csv(geneA_file, sep='\t', float_format="%.5g")

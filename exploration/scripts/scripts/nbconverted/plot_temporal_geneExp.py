
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#-------------------------------------------------------------------------------------------------------------------------------
# By Alexandra Lee (August 2018) 
#
# Plot heatmap of gene expression data as environment change from high to low oxygen levels
#
# Dataset: Pseudomonas aeruginosa gene expression compendium referenced in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5069748/
# 
# Use map_file to select only those samples from the oxygen level experiment
# 
#-------------------------------------------------------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


randomState = 123
from numpy.random import seed
seed(randomState)


# In[3]:


# load arguments
data_file = os.path.join(os.path.dirname(os.getcwd()), "data", "all-pseudomonas-gene-normalized.zip")  # repo file is zipped
map_file = os.path.join(os.path.dirname(os.getcwd()), "metadata", "mapping_oxy.txt")

PA1673like_file = os.path.join(os.path.dirname(os.getcwd()), "output", "PA1673_like_genes_v1.txt")


# In[4]:


# read in data
data = pd.read_table(data_file, header=0, sep='\t', index_col=0, compression='zip')
X = data.transpose()
X.head(5)


# In[5]:


# read in metadata file containing sample ids for dataset to consider (i.e. oxygen level experiment: E-GEOD-52445)
grp = pd.read_table(map_file, header=0, sep='\t', index_col=None)
grp


# In[6]:


# select only those rows the experiment under focus
# ordering based on timecourse experiment (high oxygen --> low oxygen)

timeline = ['maxO2', 't5', 't10', 't15', 't20', 't25', 't30', 't35', 't40', 't50', 't60', 't70', 't80', 'minO2']
dataset = pd.DataFrame()

for index, row in grp.iterrows():
    if row['Phenotype'] == timeline[index]:
        sample = str(row['Sample ID'])
        dataset = dataset.append(X[X.index.str.contains(sample, regex=False)])
        
dataset = dataset.T
dataset.shape


# In[7]:


# Heat map of all genes
plt.figure(figsize=(10, 50))
sns.heatmap(dataset[4000:5000], yticklabels=False, cmap="RdBu_r", annot=False)
#sns.clustermap(dataset)


# In[8]:


# Heat map of gene PA1673 (known to be solely regulated by Anr TF so expect it should be 
# linearly turned on as Anr turns on when oxygen levels decrease)
#sns.heatmap(dataset['PA1673'], annot=True)
plt.figure(figsize=(10, 1))
PA1673_exp = dataset[dataset.index == 'PA1673']
sns.heatmap(PA1673_exp, annot = True, cmap = "RdBu_r")


# In[9]:


# Use pearson correlation score to compare PA1673 profile with all other genes
# Select genes that have the highest 95% person correlation score as being "PA1673-like"

corr_score = []
ref_gene = np.reshape(PA1673_exp.values, (PA1673_exp.shape[1],))
for i in range(0,dataset.shape[0]):
    corr_score.append(pearsonr(ref_gene, dataset.iloc[i].values))
corr_score_df = pd.DataFrame(corr_score, index=dataset.index, columns=['Pearson', 'Pvalue'])


# In[10]:


# Select only those genes that exceed 95% quantile (i.e. PA1673-like)
threshold = corr_score_df.Pearson.quantile(q = 0.95)
PA1673_like_genes = corr_score_df.query("Pearson >= @threshold")

# control: check that PA1673 gene is in selected subset
assert("PA1673" in PA1673_like_genes.index)

type(PA1673_like_genes)
PA1673_like_genes.to_csv(PA1673like_file, sep='\t')


# In[11]:


sns.distplot(corr_score_df.Pearson)


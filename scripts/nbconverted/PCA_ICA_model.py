
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#-------------------------------------------------------------------------------------------------------------------------------
# Input: Pa gene expression data from ArrayExpress (matrix: sample x gene)
# Data compression method: PCA or ICA
# Output: Reduced Pa gene expressiond ata (matrix: sample x 2 linear combination of genes)
#-------------------------------------------------------------------------------------------------------------------------------
from sklearn.decomposition import PCA, FastICA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
np.random.seed(123)


# In[3]:


# load arguments
data_file = 'C:/Users/alexj/Documents/UPenn/CGreene/Pseudomonas/data/all-pseudomonas-gene-normalized.pcl'
map_file = 'C:/Users/alexj/Documents/UPenn/CGreene/Pseudomonas/metadata/mapping_sampleID_medium.txt'
component_num = 2

# specify either 'ica' or 'pca'
method = 'pca' 


# In[4]:


# read in data
data = pd.read_table(data_file, header = 0, sep = '\t', index_col = 0)
data = pd.DataFrame(data)
X = data.transpose()
X.head(5)
#X.shape


# In[5]:


# PCA
if method == 'pca':
    reduced = PCA(n_components=component_num)
    reduced_X = reduced.fit_transform(X)
# ICA
else:
    reduced = FastICA(n_components=component_num)
    reduced_X = reduced.fit_transform(X)


# In[6]:


# Map sample id to clinial phenotype (i.e. experimental condition)

# Note:
# According to the source (https://github.com/scikit-learn/scikit-learn/blob/14031f6/sklearn/decomposition/pca.py#L310),
# input will be transformed by np.array() before doing PCA. So row index will be lost during 
# PCA.fit_transform(X) even using a structured array or a pandas DataFrame. However, the order of the data is preserved, 
# meaning you can attach the index back afterward

X_ann = pd.DataFrame(reduced_X, index=X.index, columns=['1', '2'])

# read in mapping file (sample id --> phenotype)
map = pd.read_table(map_file, header = 0, sep = '\t', index_col = 0)

# Join 
X_new = pd.merge(X_ann, map, left_index=True, right_index=True)
X_new.head(10)
#X_new.shape


# In[7]:


# Plot
fg = sns.lmplot(x = '1', y = '2', data = X_new, hue = 'medium', fit_reg = False)
fg.add_legend()
fig_file = 'C:/Users/alexj/Documents/UPenn/CGreene/Pseudomonas/viz/'+method+'.png'
fg.fig.suptitle(method.upper()+' compressed data')
fg.savefig(fig_file)


# In[8]:


# Output compressed data
file_out = 'C:/Users/alexj/Documents/UPenn/CGreene/Pseudomonas/models/'+method+'_encoded.txt'
X_new.to_csv(file_out, sep='\t')


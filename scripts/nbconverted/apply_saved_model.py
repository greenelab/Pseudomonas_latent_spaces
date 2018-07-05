
# coding: utf-8

# In[1]:


#-------------------------------------------------------------------------------------------------------------------------------
# By Alexandra Lee (July 2018) 
#
# Apply saved model to new samples including:
#
# Encode samples from new condition using saved model
# Encode test set using saved model
# Decode estimated gene experssion after LSA
#-------------------------------------------------------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
from keras.models import model_from_json
from keras import metrics, optimizers
np.random.seed(123)


# In[2]:


# load arguments
new_data_file = os.path.join(os.path.dirname(os.getcwd()), "estimated_geneExp", "estimated_test_B.txt")
latent = True

# output files
out_file = os.path.join(os.path.dirname(os.getcwd()), "estimated_geneExp", "decoded_test_B.txt")
model_file = os.path.join(os.path.dirname(os.getcwd()), "models", "tybalt_1layer_10_trainA_decoder_model.json")
weights_file = os.path.join(os.path.dirname(os.getcwd()), "models", "tybalt_1layer_10_trainA_decoder_weights.h5")


# In[3]:


# read in data
if latent:
    new_data = pd.read_table(new_data_file, header = 0, sep = '\t', index_col = 0).transpose()
else:
    new_data = pd.read_table(new_data_file, header = 0, sep = '\t', index_col = 0)
new_data.head(5)


# In[4]:


# read in saved models

# load json and create model
json_file = open(model_file, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
                 
# load weights into new model
loaded_model.load_weights(weights_file)


# In[5]:


# Use trained model to encode new data into SAME latent space
new_reconstructed = loaded_model.predict_on_batch(new_data)
new_reconstructed_df = pd.DataFrame(new_reconstructed, index=new_data.index)


# In[6]:


# Save latent space representation
new_reconstructed_df.to_csv(out_file, sep='\t')



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
from keras.models import model_from_json, load_model
from keras import metrics, optimizers

randomState = 123
from numpy.random import seed
seed(randomState)


# In[2]:


# load arguments
input_file = os.path.join(os.path.dirname(os.getcwd()), "data","test_control.txt")
model_file = os.path.join(os.path.dirname(os.getcwd()), "models", "tybalt_1layer_10latent_encoder_model5.h5")
weights_file = os.path.join(os.path.dirname(os.getcwd()), "models", "tybalt_1layer_10latent_encoder_weights5.h5")

# output files
out_file = os.path.join(os.path.dirname(os.getcwd()), "encoded", "test_control_1layer_10latent_encoded5.txt")


# In[3]:


# read in data
data = pd.read_table(input_file, header = 0, sep = '\t', index_col = 0)
data


# In[4]:


# read in saved models

loaded_model = load_model(model_file)
# load weights into new model
loaded_model.load_weights(weights_file)


# In[5]:


# Use trained model to encode new data into SAME latent space
reconstructed = loaded_model.predict_on_batch(data)

reconstructed_df = pd.DataFrame(reconstructed, index=data.index)
reconstructed_df


# In[6]:


# Save latent space representation
reconstructed_df.to_csv(out_file, sep='\t')



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
input_file = os.path.join(os.path.dirname(os.getcwd()), "encoded", "estimated_test_control_encoded.txt")
model_file = os.path.join(os.path.dirname(os.getcwd()), "models", "tybalt_1layer_10_train_decoder_model.json")
weights_file = os.path.join(os.path.dirname(os.getcwd()), "models", "tybalt_1layer_10_train_decoder_weights.h5")

# If encoding
encoding = True

# output files
out_file = os.path.join(os.path.dirname(os.getcwd()), "output", "estimated_test_control_latent.txt")


# In[3]:


# read in data
data = pd.read_table(input_file, header = 0, sep = '\t', index_col = 0)
data


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
reconstructed = loaded_model.predict_on_batch(data)

if encoding:
    reconstructed_df = pd.DataFrame(reconstructed, index=data.index)
else:
    reconstructed_df = pd.DataFrame(reconstructed) # Can we assume the index is preserved after decoding?

reconstructed_df


# In[6]:


# Save latent space representation
reconstructed_df.to_csv(out_file, sep='\t')


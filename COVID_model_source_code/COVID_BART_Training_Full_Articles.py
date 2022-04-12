#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip install simpletransformers')
import os
import json
import joblib
import time

import pandas as pd
import random
from IPython.display import display, HTML


import torch
import transformers
from transformers import BartTokenizer, BartForConditionalGeneration
from simpletransformers.ner import NERModel, NERArgs

#connect to hugging face
# get_ipython().system('git clone https://github.com/huggingface/transformers && cd transformers install')
# get_ipython().system('pip install -q ./transformers')

# get_ipython().system('pip install simpletransformers')
# get_ipython().system('pip install transformers')
# get_ipython().system('pip install ipywidgets')

from simpletransformers.seq2seq import Seq2SeqModel,Seq2SeqArgs


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using: ', device)


path_models = './models/'
path_outputs = './outputs/'
path_processed_data= './processed_data/'
#path_covid_ner_model = './models/covid_ner_model_100k/'


# In[5]:


jsonFile = open(os.path.join(path_processed_data, 'covid_train_ner_v3.json'), "r")

jsonContent = jsonFile.read()
details_dict = json.loads(jsonContent)
jsonFile.close()

#Log: X_train, Y_train, X_valid etc sizes:  288000 288000 16000 16000 16000 16000

Y_train = details_dict['headlines'][0:25000]
X_train = details_dict['text'][0:25000]

jsonFile = open(os.path.join(path_processed_data, 'covid_valid_ner_v3.json'), "r")
jsonContent = jsonFile.read()
details_dict = json.loads(jsonContent)
jsonFile.close()

Y_valid = details_dict['headlines'][0:5000]
X_valid = details_dict['text'][0:5000]


jsonFile = open(os.path.join(path_processed_data, 'covid_test_ner_v3.json'), "r")
jsonContent = jsonFile.read()
details_dict = json.loads(jsonContent)
jsonFile.close()

Y_test = details_dict['headlines'][0:5000]
X_test = details_dict['text'][0:5000]


# In[6]:


print(X_train[0])



# In[7]:


print("Log: X_train, Y_train, X_valid etc sizes: ", len(X_train), len(Y_train), len(X_valid), len(Y_valid), len(X_test), len(Y_test))


# In[8]:


X_train[1:2]


# In[9]:


Y_train[1:2]


# In[10]:


details = {
    'headlines' : Y_train,
    'text' : X_train
}

df_train = pd.DataFrame(details)


# In[11]:


details = {
    'headlines' : Y_valid,
    'text' : X_valid
}

df_valid = pd.DataFrame(details)


# In[12]:


details = {
    'headlines' : Y_test,
    'text' : X_test
}

df_test = pd.DataFrame(details)


# In[13]:


tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using: ', torch_device)


# In[14]:


# Rename columns as per pretrained model required format
df_train=df_train.rename(columns={'headlines':'target_text','text':'input_text'})
df_valid=df_valid.rename(columns={'headlines':'target_text','text':'input_text'})
df_test=df_test.rename(columns={'headlines':'target_text','text':'input_text'})
#df = {dataset, target}
model_args = Seq2SeqArgs()
# Initializing number of epochs
#model_args.num_train_epochs = 25
model_args.num_train_epochs = 10
# Initializing no_save arg
model_args.no_save = True
model_args.overwrite_output_dir = True

# Initializing evaluation args
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True


# In[15]:


df_train['target_text'][0]


# In[16]:


df_train['input_text'][0]


# In[17]:


df_valid['target_text'][0]


# In[18]:


df_valid['input_text'][0]


# In[19]:


df_test['target_text'][0]


# In[20]:


df_test['input_text'][0]


# In[21]:


model = Seq2SeqModel(
encoder_decoder_type="bart",
encoder_decoder_name="facebook/bart-large",
args=model_args,
use_cuda=True,
)
# Splitting data into train-test
# from sklearn.model_selection import train_test_split
# train_df, test_df = train_test_split(df, test_size=0.2)


# In[22]:


df_train.shape, df_valid.shape, df_test.shape


# In[32]:


batch_size_train = 10000
batch_size_test = batch_size_train//4
number_of_batches = 25000//batch_size_train

number_of_batches

#len(df_train)//12000
toy_size = 20
full_size = 3000
batch_size_train = full_size

batch_size_test = batch_size_train//4
number_of_batches = len(df_train)//batch_size_train

print("Log: number_of_batches: ",number_of_batches)


# In[23]:


def current_time_min():
    time_in_mins = time.time()/60.0
    return round(time_in_mins, 4)


# In[ ]:


# Training the model and keeping eval dataset as test data
# 16000 for aws, 800 for local VM


count = 0
epocs = model_args.num_train_epochs
acc_time = 0

#change these:

# df_valid = df_valid[:8000]
# df_test = df_test[:8000]



for i in range(number_of_batches):
    start_time = current_time_min()
    count += 1
    start_offset_train = i * batch_size_train
    #start_offset_test = i * batch_size_test
    end_offset_train = start_offset_train + batch_size_train
    #end_offset_test = start_offset_test + batch_size_test
    model.train_model(df_train[start_offset_train:end_offset_train], eval_data=df_valid)
    model_filename = "covid_full_bart" + str(count) + "_" + str(start_offset_train) + "_" + str(end_offset_train) + "_" + str(epocs) + "e.pkl"
    joblib.dump(model, os.path.join(path_models, model_filename))
    results = model.eval_model(df_test)
    batch_time = current_time_min() - start_time
    acc_time = acc_time + batch_time/60
    randint = random.randrange(1, len(df_test), 3)
    results_json = {
        'iteration' : count,
        'start' : start_offset_train,
        'end' : end_offset_train,
        'eval_loss' : results["eval_loss"],
        'batch_time' : batch_time,
        'acc_time' : acc_time
    }
    jsonString = json.dumps(results_json)
    results_filename = "results_" + str(start_offset_train) + "_" + str(end_offset_train) + "_" + str(count)
    results_file = open(os.path.join(path_models, "results_covid_full_bart.txt"), "a")
    results_file.write(jsonString + '\n')
    print(jsonString)
    results_file.close()
    print("Count: ", count, "Results: ", results)


# In[26]:


# Generating summaries on news test data
#results = model.eval_model(df_test[100:200])
results = model.eval_model(df_test)

# print the loss
results


# In[ ]:









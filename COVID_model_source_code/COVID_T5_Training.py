#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, psutil  
import json
import joblib
import time

import pandas as pd
import random
from IPython.display import display, HTML
import torch
import numpy as np


# In[2]:


path_models = './models/'
path_outputs = './outputs/'
path_processed_data= './processed_data/'


# In[26]:



jsonFile = open(os.path.join(path_processed_data, 'covid_train_ner_v3.json'), "r")
jsonContent = jsonFile.read()
details_dict = json.loads(jsonContent)
jsonFile.close()

X_train = details_dict['text'][0:73280]
#X_train = details_dict['text_ner']
Y_train = details_dict['headlines'][0:73280]

jsonFile = open(os.path.join(path_processed_data, 'covid_valid_ner_v3.json'), "r")
jsonContent = jsonFile.read()
details_dict = json.loads(jsonContent)
jsonFile.close()

X_valid = details_dict['text'][0:9160]
#X_valid = details_dict['text_ner']
Y_valid = details_dict['headlines'][0:9160]

jsonFile = open(os.path.join(path_processed_data, 'covid_test_ner_v3.json'), "r")
jsonContent = jsonFile.read()
details_dict = json.loads(jsonContent)
jsonFile.close()

X_test = details_dict['text'][0:9160]
#X_test = details_dict['text_ner']
Y_test = details_dict['headlines'][0:9160]

#((73280, 3), (9160, 3), (9160, 3))

details = {
    'headlines' : Y_train,
    'text' : X_train
}

df_train = pd.DataFrame(details)

details = {
    'headlines' : Y_valid,
    'text' : X_valid
}

df_valid = pd.DataFrame(details)

details = {
    'headlines' : Y_test,
    'text' : X_test
}

df_test = pd.DataFrame(details)


# In[27]:


print(len(X_train), len(Y_train), len(X_valid), len(Y_valid), len(X_test), len(Y_test))


# In[28]:


def cpu_stats():
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2. ** 30
    return 'memory GB:' + str(np.round(memory_use, 2))

def current_time_min():
    time_in_mins = time.time()/60.0
    return round(time_in_mins, 4)


# In[29]:


cpu_stats()


# In[30]:


df_train['prefix'] = "summarize"
df_test['prefix'] = "summarize"
df_valid['prefix'] = "summarize"

model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 512,
    "train_batch_size": 16,
    "num_train_epochs": 4,
}


# In[31]:


# Rename columns as per pretrained model required format
df_train=df_train.rename(columns={'headlines':'target_text','text':'source_text'})
df_valid=df_valid.rename(columns={'headlines':'target_text','text':'source_text'})
df_test=df_test.rename(columns={'headlines':'target_text','text':'source_text'})

#((73280, 3), (9160, 3), (9160, 3))


# In[32]:


df_train.shape, df_valid.shape, df_test.shape


# In[33]:



# In[30]:


from simplet5 import SimpleT5


# In[31]:


model = SimpleT5()
model.from_pretrained(model_type="t5", model_name="t5-base")
#model.from_pretrained(model_type="t5", model_name="t5-large")


# In[33]:


model.train(train_df=df_train,
            eval_df=df_valid, 
            source_max_token_len=128, 
            target_max_token_len=50, 
            batch_size=16, 
            max_epochs=10, 
            use_gpu=True
           )


# In[34]:


# model_filename = "model_covid_t5_base.pkl" 
# joblib.dump(model, os.path.join(path_models, model_filename))
    
    


# In[26]:


df_test.head()


# In[22]:


#model = joblib.load(os.path.join(path_models, 'model_t5_3e.pkl'))
#model.load_model("t5", "outputs/simplet5-epoch-2-train-loss-2.6228-val-loss-2.8411.1", use_gpu=True)
model.load_model("t5", "outputs/simplet5-epoch-0-train-loss-1.539-val-loss-3.7824", use_gpu=True)
#model.load_model("t5", "model_covid_t5_base.pkl", use_gpu=True)
# model = joblib.load(os.path.join(path_models, 'model_covid_t5_base.pkl'))


# In[23]:


t5_prepared_Text = "summarize: " + df_test[0:2]['text'][1]
t5_prepared_Text


# In[24]:


P1 = model.predict(t5_prepared_Text)








###########################



###########################















# # In[25]:


# P1


# # In[31]:


# data_name = 'cnn_test'
# input_filename = data_name + '_summary_ner_v3.json' 
# output_filename = data_name + '_prediction_t5_ner_v3.json'


# # In[32]:


# jsonFile = open(os.path.join(path_processed_data, input_filename), "r")
# jsonContent = jsonFile.read()
# details_dict = json.loads(jsonContent)
# jsonFile.close()

# details = {
#     'headlines' : details_dict['headlines'],
#     'text' : details_dict['text'],
#     'keywords' : details_dict['keywords'],
#     'summary_art' : details_dict['summary_art'],
#     'entities' : details_dict['entities'],
#     'text_ner' : details_dict['text_ner'],
#     'sentence_1s' : details_dict['sentence_1s'],
#     'sentence_3s' : details_dict['sentence_3s'],
#     'summary_ext' : details_dict['summary_ext'],
#     'summary_abs' : details_dict['summary_abs'],
#     'summary_extabs' : details_dict['summary_extabs'],
#     'predict_1s' : details_dict['predict_1s'],
#     'predict_3s' : details_dict['predict_3s'],
#     'predict_text' : details_dict['predict_text'],
#     'predict_ext' : details_dict['predict_ext'],
#     'predict_abs' : details_dict['predict_abs'],
#     'predict_extabs' : details_dict['predict_extabs'],
#     'predict_art' : details_dict['summary_art']
# }


# df_test = pd.DataFrame(details)


# # In[33]:


# df = df_test

# def predict_t5(row):
    
#     predict_3s = []
#     predict_text = []
#     predict_ext = []
#     predict_abs = []    
#     predict_extabs = []
#     predict_art = []    
    
    
#     try:
#         t5_text = "summarize: " + row[1]['sentence_3s']
#         predict_3s = model.predict(t5_text)[0]
        
#         t5_text = "summarize: " + row[1]['text']
#         predict_text = model.predict(t5_text)[0]
        
#         t5_text = "summarize: " + row[1]['summary_ext']
#         predict_ext = model.predict(t5_text)[0]
        
#         t5_text = "summarize: " + row[1]['summary_abs']
#         predict_abs = model.predict(t5_text)[0]
        
#         t5_text = "summarize: " + row[1]['summary_extabs']
#         predict_extabs = model.predict(t5_text)[0]
        
#         t5_text = "summarize: " + row[1]['summary_art']
#         predict_art = model.predict(t5_text)[0]
        
#     except Exception as error:
#         print("Exception in predict_t5():", error)
#         return predict_3s, predict_text, predict_ext, predict_abs, predict_extabs, predict_art
#     return predict_3s, predict_text, predict_ext, predict_abs, predict_extabs, predict_art 


# # In[34]:


# from joblib import Parallel, delayed
# start = time.time()

# df = df.reset_index()

# S = Parallel(n_jobs=1)(delayed(predict_t5)(row) for row in df.iterrows())

# print("Response time (mins): ", round((time.time() - start)/60, 2))


# # In[35]:


# predict_3s = []
# predict_text = []
# predict_ext = []
# predict_abs = []    
# predict_extabs = []
# predict_art = [] 

# S = list(filter(None, S))
# for i in range(len(S)):  
#         predict_3s.append(S[i][0])
#         predict_text.append(S[i][1])  
#         predict_ext.append(S[i][2])
#         predict_abs.append(S[i][3])
#         predict_extabs.append(S[i][4])
#         predict_art.append(S[i][5])   
        


# # In[36]:


# details = {
#     'headlines' : details_dict['headlines'],
#     'text' : details_dict['text'],
#     'keywords' : details_dict['keywords'],
#     'summary_art' : details_dict['summary_art'],
#     'entities' : details_dict['entities'],
#     'sentence_1s' : details_dict['sentence_1s'],
#     'sentence_3s' : details_dict['sentence_3s'],
#     'summary_ext' : details_dict['summary_ext'],
#     'summary_abs' : details_dict['summary_abs'],
#     'summary_extabs' : details_dict['summary_extabs'],
#     'summary_ner' : details_dict['summary_ner'],
#     'summary_t5' : details_dict['summary_t5'],
#     'predict_1s' : details_dict['sentence_1s'],
#     'predict_3s' : predict_3s,
#     'predict_text' : predict_text,
#     'predict_ext' : predict_ext,
#     'predict_abs' : predict_abs,
#     'predict_extabs' : predict_extabs,
#     'predict_art' : predict_art,
# }


# jsonString = json.dumps(details)
# jsonFile = open(os.path.join(path_processed_data, output_filename), "w")
# jsonFile.write(jsonString)
# jsonFile.close()


# # In[ ]:





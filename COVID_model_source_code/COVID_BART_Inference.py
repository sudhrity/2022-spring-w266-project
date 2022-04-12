#!/usr/bin/env python
# coding: utf-8

# In[42]:


import os, psutil  
import json
import joblib
import time

import pandas as pd
import random
from IPython.display import display, HTML
import torch
import numpy as np



path_models = './models/'
path_outputs = './outputs/'
path_processed_data= './processed_data/'


# In[45]:



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


# In[46]:


print(len(X_train), len(Y_train), len(X_valid), len(Y_valid), len(X_test), len(Y_test))


# In[47]:


def cpu_stats():
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2. ** 30
    return 'memory GB:' + str(np.round(memory_use, 2))

def current_time_min():
    time_in_mins = time.time()/60.0
    return round(time_in_mins, 4)


# In[48]:


cpu_stats()


# In[49]:


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


# In[50]:


# Rename columns as per pretrained model required format
df_train=df_train.rename(columns={'headlines':'target_text','text':'source_text'})
df_valid=df_valid.rename(columns={'headlines':'target_text','text':'source_text'})
df_test=df_test.rename(columns={'headlines':'target_text','text':'source_text'})

#((73280, 3), (9160, 3), (9160, 3))


# In[51]:


df_train.shape, df_valid.shape, df_test.shape


# In[52]:


#get_ipython().system('pip install simplet5')


# In[53]:


from simplet5 import SimpleT5


# In[54]:


model = SimpleT5()
model.from_pretrained(model_type="t5", model_name="t5-base")
#model.from_pretrained(model_type="t5", model_name="t5-large")



df_test.head()


# In[58]:

#Load the full T5 model
model.load_model("t5", "models/t5_covid_base/simplet5-epoch-9-train-loss-1.0452-val-loss-0.6521", use_gpu=True)

#model = joblib.load(os.path.join(path_models, 'model_t5_3e.pkl'))
#model.load_model("t5", "outputs/simplet5-epoch-2-train-loss-2.6228-val-loss-2.8411.1", use_gpu=True)
#model.load_model("t5", "model_covid_t5_base.pkl", use_gpu=True)
# model = joblib.load(os.path.join(path_models, 'model_covid_t5_base.pkl'))


# In[59]:


t5_prepared_Text = "summarize: " + df_test[0:5]['source_text'][4]
t5_prepared_Text


# In[60]:


P1 = model.predict(t5_prepared_Text)


# In[61]:


P1


# In[62]:


data_name = 'covid_test'
#input_filename = data_name + '_summary_ner_v3.json' 
input_filename = 'covid_filtered_ner_test_v3_prediction.json' 
output_filename = data_name + '_prediction_t5_bart_v3.json'


# In[63]:


import re
from nltk.tokenize import sent_tokenize, word_tokenize


jsonFile = open(os.path.join(path_processed_data, input_filename), "r")
jsonContent = jsonFile.read()
details_dict = json.loads(jsonContent)
jsonFile.close()

#Create the data for 3s - 1st 3 sentences of an article
textArr = details_dict['text']

sentence_3s = []
# for i,j in enumerate(textArr):
#     #j = "The dog was called Wellington. It belonged to Mrs. Shears who was our friend. She lived on the opposite side of the road, two houses to the left."

#     sen = ' '.join(re.split(r'(?<=[.])\s', j)[:3])  
#     sentence_3s.append(sen)
# print(sentence_3s[3:4])
    

sentence_3s = []

for i,j in enumerate(textArr):
    #j = "The dog was called Wellington. It belonged to Mrs. Shears who was our friend. She lived on the opposite side of the road, two houses to the left."

    sen = sent_tokenize(j)
    sen1 = ' '.join(sen[:3])
    sentence_3s.append(sen1)

print ("..................")
print(sentence_3s[3:4])

print ("..................")
print(len(sentence_3s))
    
#    if i%100 ==0:
#        print(split_string[0:2])
#        print ("......")
    
    
#     first_sentence = []
#     for ele in split_string:
#         first_sentence.append(ele)
#         if ele in linking_verbs:
#             linked_verb_booly = True
#         if '.' in ele and linked_verb_booly == True:
#             break


# In[64]:


details = {
    'headlines' : details_dict['headlines'],
    'text' : details_dict['text'],
    'ner_filtered_text' : details_dict['ner_filtered_text'],
    'sentence_3s' : sentence_3s

#     'predict_base_bart' : details_dict['predict_base'],
#     'predict_bart_ner_fullTrained' : details_dict['predict_ner_fullTrained'],
#     'predict_bart_ner_nerTrained' : details_dict['predict_ner_nerTrained'],
#     'predict_bart_full_nerTrained' : details_dict['predict_full_nerTrained'],
    

}


df_test = pd.DataFrame(details)


# In[65]:


df = df_test

def predict_t5(row):
    
    predict_3s = []
    predict_no_filter = []
    predict_ner_filter = []
   
    
    try:
        t5_text = "summarize: " + row[1]['sentence_3s']
        predict_3s = model.predict(t5_text)[0]
  
        t5_text = "summarize: " + row[1]['text']
        predict_no_filter = model.predict(t5_text)[0] 
        
        t5_text = "summarize: " + row[1]['ner_filtered_text']
        predict_ner_filter = model.predict(t5_text)[0]
        
     
    except Exception as error:
        print("Exception in predict_t5():", error)
        return predict_3s, predict_no_filter, predict_ner_filter
    return predict_3s, predict_no_filter, predict_ner_filter 


# In[73]:


from joblib import Parallel, delayed
start = time.time()

#df = df.reset_index()
S=[]

#S = Parallel(n_jobs=1)(delayed(predict_t5)(row) for row in df.iterrows())

for i, row in enumerate(df.iterrows()):
    S.append(predict_t5(row))
    
    if i%100 == 0:
        print ("Log: base t5 predict ", i);
                       

print("Log T5 Base Response time (mins): ", round((time.time() - start)/60, 2))
#print(pd.DataFrame(S))


# In[74]:


predict_t5_3s = []
predict_t5_no_filter = []
predict_t5_ner_filter = []


S = list(filter(None, S))
for i in range(len(S)):  
        predict_t5_3s.append(S[i][0])
        predict_t5_no_filter.append(S[i][1])  
        predict_t5_ner_filter.append(S[i][2])
       


# In[75]:





# In[31]:


#COVID NER filtered and trained model
model.load_model("t5", "models/t5_covid_ner/simplet5-epoch-9-train-loss-1.0844-val-loss-2.0962", use_gpu=True)
#print(df_test)

t5_prepared_Text = "summarize: " + df_test[0:5]['text'][4]
t5_prepared_Text
#P1 = model.predict(t5_prepared_Text)


# In[41]:


# from joblib import Parallel, delayed
# start = time.time()
# df = df.reset_index()
# S = Parallel(n_jobs=1)(delayed(predict_t5)(row) for row in df.iterrows())
# print("Response time (mins): ", round((time.time() - start)/60, 2))

from joblib import Parallel, delayed
start = time.time()

#df = df.reset_index()
S=[]

#S = Parallel(n_jobs=1)(delayed(predict_t5)(row) for row in df.iterrows())

for i, row in enumerate(df.iterrows()):
    S.append(predict_t5(row))
    
    if i%100 == 0:
        print ("Log: Ner filter trained t5 predict ", i);
                        

print("Log: Response time (mins): ", round((time.time() - start)/60, 2))


predict_t5_full_nerTrained = []
predict_t5_3s_nerTrained = []
predict_t5_ner_nerTrained = []


S = list(filter(None, S))
for i in range(len(S)):  
        predict_t5_3s_nerTrained.append(S[i][0])  
        predict_t5_full_nerTrained.append(S[i][1])
        predict_t5_ner_nerTrained.append(S[i][2])


# In[36]:




details = {
    'headlines' : details_dict['headlines'],
    'text' : details_dict['text'],
    'sentence_3s' : sentence_3s,
    'ner_filtered_text' : details_dict['ner_filtered_text'],
    
    'predict_base_bart_fullTrained' : details_dict['predict_base'],
    'predict_bart_ner_fullTrained' : details_dict['predict_ner_fullTrained'],
    'predict_bart_ner_nerTrained' : details_dict['predict_ner_nerTrained'],
    'predict_bart_full_nerTrained' : details_dict['predict_full_nerTrained'],
  
   
    'predict_base_t5_full_fullTrained' : predict_t5_no_filter,
    'predict_t5_3s_fullTrained' : predict_t5_3s,
    'predict_t5_ner_fullTrained' : predict_t5_ner_filter,
    
    'predict_t5_full_nerTrained' : predict_t5_full_nerTrained,
    'predict_t5_3s_nerTrained' : predict_t5_3s_nerTrained,
    'predict_t5_ner_nerTrained' : predict_t5_ner_nerTrained
}

jsonString = json.dumps(details)
jsonFile = open(os.path.join(path_processed_data, output_filename), "w")
jsonFile.write(jsonString)
jsonFile.close()


# In[40]:


#input_filename = data_name + '_summary_ner_v3.json' 
input_filename = data_name + '_prediction_t5_bart_v3.json'

jsonFile = open(os.path.join(path_processed_data, input_filename), "r")
jsonContent = jsonFile.read()
details_dict = json.loads(jsonContent)
jsonFile.close()

X = details_dict['text']
Y = details_dict['headlines']
S1 = details_dict['ner_filtered_text']

P1 = details_dict['predict_bart_ner_fullTrained']
P2 = details_dict['predict_bart_ner_nerTrained']
P3 = details_dict['predict_bart_full_nerTrained']

P4 = details_dict['predict_base_t5_full_fullTrained']
P5 = details_dict['predict_t5_3s_fullTrained']
P6 = details_dict['predict_t5_ner_fullTrained']
P7 = details_dict['predict_t5_full_nerTrained']
P8 = details_dict['predict_t5_3s_nerTrained']
P9 = details_dict['predict_t5_ner_nerTrained']

    
details = {
    'text' : X,
    'headlines' : Y,
    'ner_filtered_text' : S1,
    
    'predict_bart_ner_fullTrained' : P1,
    'predict_bart_ner_nerTrained' : P2,
    'predict_bart_full_nerTrained' : P3,
    
    'predict_base_t5_full_fullTrained' : P4,
    'predict_t5_3s_fullTrained' : P5,
    'predict_t5_ner_fullTrained' : P6,
    'predict_t5_full_nerTrained' : P7,
    'predict_t5_3s_nerTrained': P8,
    'predict_t5_ner_nerTrained' : P9
}

df_score = pd.DataFrame(details)

print(df_score)


# In[ ]:





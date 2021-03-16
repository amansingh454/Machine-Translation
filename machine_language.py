#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import nltk
import numpy as np
from tqdm import tqdm
import string
import nltk


# In[2]:


import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import one_hot
from keras.layers import Dense,ReLU,Activation,Dropout
from keras.layers import LSTM,Bidirectional,Embedding
from keras.models import Model,Sequential,Input
from keras.preprocessing.sequence import pad_sequences


# In[3]:


nltk.download()


# In[4]:


with open(r'C:\Users\aman\programming\ml projects\dataset\deu-eng\deu.txt',"r",encoding="latin-1")as fp:
    text_file=fp.readlines()


# In[ ]:





# In[5]:


eng_sent=[]
eng_unique_word=[]
spanish_sent=[]
spanish_unique_word=[]


# In[6]:


s='Hello!\tHallo!\tCC-BY 2.0 (France) Attribution: tatoeba.org #373330 (CK) & #380701 (cburgmer)\n'
s.split('\t')[0]


# In[7]:


for i in range(50000):
    index=np.random.randint(len(text_file))
    eng_text=text_file[index].split("\t")[0]
    spanish_text=text_file[index].split("\t")[1]
    eng_text=re.sub("[0-9]"," ",eng_text)
    spanish_text=re.sub("[0-9]"," ",spanish_text)
    eng_text=re.sub("\s+"," ",eng_text)
    spanish_text=re.sub("\s+"," ",spanish_text)
    eng_text=eng_text.lower()
    spanish_text=spanish_text.lower()
    eng_sent.append(eng_text)
    spanish_sent.append(spanish_text)    


# In[8]:


print(eng_sent[:2])
spanish_sent[:2]


# In[9]:


len(eng_sent)


# In[10]:


a=[]
b=[]


# In[11]:


vocab_size=50000


# In[12]:


for i in range(50000):
    a.append(one_hot(eng_sent[i],n=vocab_size))
    b.append(one_hot(spanish_sent[i],n=vocab_size))


# In[13]:


a[3]


# In[14]:


print(a[:2])
b[:2]


# In[15]:


z=[]
for i in range(50000):
    z.append(len(a[i]))
max(z)


# In[16]:


max_len=15
a=pad_sequences(a,maxlen=max_len,padding='pre')


# In[17]:


print(a)


# In[18]:


b=pad_sequences(b,maxlen=max_len,padding='pre')
b

def encoder_decoder():
    emb_layer=Embedding(vocab_size,20,input_length=max_len,mask_zero=True)
    encoder_input=Input(shape=(max_len,))
    emb_layer=emb_layer(encoder_input)
    lstm_encoder=LSTM(units=15,activation='relu',return_state=True)
    lstm_1,state_h,state_c=lstm_encoder(emb_layer)
    initial_state=[state_h,state_c]
    model=Model(encoder_input,state_h,state_c)
    
    decoder_input=Input(shape=(None,max_len))
    lstm=LSTM(units=15,activation='relu',return_sequences=True,return_state=True)
    decoder_output, decoder_state_h, decoder_state_c=lstm(decoder_input,initial_state)
    decoder_dense=Dense(units=max_len,activation='tanh')
    decoder_dense_output=decoder_dense(decoder_output)
    
    model=Model([encoder_input,decoder_input],decoder_dense_output)
 
    encoder_model = Model(encoder_input, initial_state)
    # define inference decoder
    
    decoder_state_input_h = Input(shape=(max_len,))
    decoder_state_input_c = Input(shape=(max_len,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = lstm(decoder_input, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_input] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    # return all models
    return model, encoder_model, decoder_model
# In[119]:


def encoder_decoder():
    emb_layer=Embedding(vocab_size,20,input_length=max_len,mask_zero=True)
    encoder_input=Input(shape=(max_len,))
    emb_layer=emb_layer(encoder_input)
    lstm_encoder=LSTM(units=15,activation='relu',return_state=True)
    lstm_1,state_h,state_c=lstm_encoder(emb_layer)
    initial_state=[state_h,state_c]
    
    decoder_input=Input(shape=(max_len,))
    lstm=LSTM(units=15,activation='relu',return_sequences=True,return_state=True)
    decoder_output, decoder_state_h, decoder_state_c=lstm(decoder_input,initial_state)
    decoder_dense=Dense(units=max_len,activation='tanh')
    decoder_dense_output=decoder_dense(decoder_output)
    
    model=Model([encoder_input,decoder_input],decoder_dense_output)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit([encoder_input, decoder_input], decoder_dense_output,epochs=10,validation_split=0.2)



    encoder_model = Model(encoder_input, initial_state)
    # define inference decoder
    
    decoder_state_input_h = Input(shape=(max_len,))
    decoder_state_input_c = Input(shape=(max_len,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = lstm(units=15,batch_size=64)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_input] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    # return all models
    return model, encoder_model, decoder_model


# In[120]:


x=encoder_decoder()
print(x[0],x[1],x[2])


# In[ ]:


LSTM()


# In[84]:


x[0].summary()


# In[85]:


x[1].summary()


# In[86]:


x[2].summary()


# In[87]:





# In[ ]:


x[0].fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)


# In[56]:


x[1].compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])


# In[57]:


def predict(input_data,timestamp,encoder,decoder,cardinality):
    initial_state=encoder.predict(input_data.reshape(1,15))
    output=list()
    target_seq = np.array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
    for i in range(timestamp):
        yhat,h1,c1=decoder.predict([target_seq]+initial_state)
        y=np.argmax(yhat[0,0,:])
        output.append(y)
        initial_state=[h1,c1] 
        target_seq=yhat
    return np.array(output)


# In[58]:


x=predict(a[3],15,x[1],x[2],15)
print(x)


# In[ ]:





# In[ ]:





# In[ ]:





# In[27]:


input1=Input(shape=(3,1))
lstm_1,h,c=LSTM(3,activation='relu',return_state=True)(input1)
model=Model(input1,output=[lstm_1,h,c])
data=np.array([[0.1,0.2,0.3],[0.8,0.9,0.3]]).reshape(2,3,1)
model.predict(data)


# In[ ]:





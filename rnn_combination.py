import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re
def remove_at(text):
    txt=re.sub("@[a-zA-Z0-9]+", "", text)
    return txt

messages=pd.read_csv('datasets/Usairline/tweets_clean.csv')
messages['no_at']=messages['text'].apply(lambda x:remove_at(x))
print(messages.head())
labels=messages['label'].astype('int32')
X_train,X_test,y_train,y_test=train_test_split(messages['no_at'],messages['label'],test_size=0.2)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#Initialize and fit the tokenizer
tokenizer=Tokenizer()
tokenizer.fit_on_texts(X_train)#makes av vocabulary with an index number for every word

#Use the tokenizer to tranform the train and test sets
X_train_seq=tokenizer.texts_to_sequences(X_train)#replaces the words with their indexes
X_test_seq=tokenizer.texts_to_sequences(X_test)


#Padding
X_train_seq_padded=pad_sequences(X_train_seq,50)
X_test_seq_padded=pad_sequences(X_test_seq,50)


import keras.backend as K
from keras.layers import Dense,Embedding,LSTM
from keras.models import Sequential

def recall_m(y_true,y_pred):
    true_positives=K.sum(K.round(K.clip(y_true*y_pred,0,1)))
    possible_positives=K.sum(K.round(K.clip(y_true,0,1)))
    recall=true_positives/(possible_positives+K.epsilon())
    return recall

def precision_m(y_true,y_pred):
    true_positives=K.sum(K.round(K.clip(y_true*y_pred,0,1)))
    predicted_positives=K.sum(K.round(K.clip(y_pred,0,1)))
    precision=true_positives/(predicted_positives+K.epsilon())
    return precision

#Create Model
model=Sequential()
model.add(Embedding(len(tokenizer.index_word)+1,64))
model.add(LSTM(64,dropout=0,recurrent_dropout=0))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
print(model.summary())

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy',precision_m,recall_m])


history=model.fit(X_train_seq_padded,y_train,batch_size=64,epochs=5,validation_data=(X_test_seq_padded,y_test))


import matplotlib.pyplot as plt
for i in ['accuracy','precision_m','recall_m']:
    acc=history.history[i]
    val_acc=history.history['val_{}'.format(i)]
    epochs=range(1,len(acc)+1)
    plt.figure()
    plt.plot(epochs,acc,label='Training ' +i)
    plt.plot(epochs,val_acc,label='validation '+i)
    plt.legend()
    plt.show()
"""
with 32 shaped rnn and softmax in last layer
 combination.csv loss: 7.6457 - accuracy: 0.5014 - precision_m: 0.5014 - recall_m: 1.0000 - val_loss: 7.3751 - val_accuracy: 0.5164 - val_precision_m: 0.5231 - val_recall_m: 1.0000
"""
"""
with 64 shaped rnn and sigmoid in last layer
 combination.csv loss: 0.0581 - accuracy: 0.9877 - precision_m: 0.9913 - recall_m: 0.9843 - val_loss: 0.5946 - val_accuracy: 0.8036 - val_precision_m: 0.8588 - val_recall_m: 0.7805
"""

"""
stanford with 32 shaped rnn and softmax in last layer

 - loss: 6.9573 - accuracy: 0.5463 - precision_m: 0.5463 - recall_m: 1.0000forrtl: error (200): program aborting due to
"""
"""
stanford with 64 shaped rnn and softmax in last layer
 103s 1ms/step - loss: 6.9663 - accuracy: 0.5457 - precision_m: 0.5457 - recall_m: 1.0000 - val_loss: 6.9180 - val_accuracy: 0.5463 - val_precision_m: 0.5463 - val_recall_m: 1.0000

"""
"""
stanford with 64 shaped rnn and sigmoid in last layer

1ms/step - loss: 0.1206 - accuracy: 0.9530 - precision_m: 0.9550 - recall_m: 0.9590 - val_loss: 0.2675 - val_accuracy: 0.9139 - val_precision_m: 0.9110 - val_recall_m: 0.9336
"""
"""
tweets with 32 shaped rnn and softmax in last layer

tep - loss: 12.2440 - accuracy: 0.2015 - precision_m: 0.2013 - recall_m: 1.0000 - val_loss: 11.9273 - val_accuracy: 0.2178 - val_precision_m: 0.2153 - val_recall_m: 0.9863
"""
"""
tweets with 64 shaped rnn and sigmoid in last layer
- 5s 561us/step - loss: 0.0263 - accuracy: 0.9922 - precision_m: 0.9839 - recall_m: 0.9794 - val_loss: 0.3782 - val_accuracy: 0.9095 - val_precision_m: 0.8167 - val_recall_m: 0.6828
"""

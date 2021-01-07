import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
def join_row(thing):
    row=''.join([word for word in thing])
    return row
pd.set_option('display.max_colwidth',1000)
messages=pd.read_csv('datasets/stanfordSentimentTreebank/stanford_fixed2.csv')
messages=messages[messages['label'].isin([1.0,0.0])]
print(messages.head())
print(messages['body_text'].isnull().sum())
messages.dropna(inplace=True)
print(messages['body_text'].isnull().sum())
print(messages['label'].value_counts())
labels=messages['label'].astype('int32')
X_train,X_test,y_train,y_test=train_test_split(messages['body_text'],labels,test_size=0.2)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#Initialize and fit the tokenizer
tokenizer=Tokenizer()
tokenizer.fit_on_texts(X_train)#makes av vocabulary with an index number for every word

#Use the tokenizer to tranform the train and test sets
X_train_seq=tokenizer.texts_to_sequences(X_train)#replaces the words with their indexes
X_test_seq=tokenizer.texts_to_sequences(X_test)


#Padding
X_train_seq_padded=pad_sequences(X_train_seq,100)
X_test_seq_padded=pad_sequences(X_test_seq,100)


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
    plt.plot(epochs,acc,label='Training' +i)
    plt.plot(epochs,val_acc,label='validation '+i)
    plt.legend()
    plt.show()
# combination.csv loss: 7.6457 - accuracy: 0.5014 - precision_m: 0.5014 - recall_m: 1.0000 - val_loss: 7.3751 - val_accuracy: 0.5164 - val_precision_m: 0.5231 - val_recall_m: 1.0000
"""
stanford with 32 shaped rnn and softmax in last layer

 - loss: 6.9573 - accuracy: 0.5463 - precision_m: 0.5463 - recall_m: 1.0000forrtl: error (200): program aborting due to
"""
"""
stanford with 64 shaped rnn and softmax in last layer
 103s 1ms/step - loss: 6.9663 - accuracy: 0.5457 - precision_m: 0.5457 - recall_m: 1.0000 - val_loss: 6.9180 - val_accuracy: 0.5463 - val_precision_m: 0.5463 - val_recall_m: 1.0000

"""

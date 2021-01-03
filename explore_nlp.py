import pandas as pd
import string
import re
import nltk
import os
stopwords=nltk.corpus.stopwords.words('english')



all_files=[]
for dirname, _, filenames in os.walk('datasets/Combination'):
    for filename in filenames:
        all_files.append(filename)
print(all_files)
df=pd.concat(all_files)
print(df.head())
"""
df=pd.read_csv("datasets/Combination/yelp_labelled.txt",sep='\t',header=None)
df.columns=['body_text','label']
print(df.head())
def clean_text(text):
    text=''.join([char for char in text if char not in string.punctuation])
    tokens=re.split('\W+',text)
    text=[word for word in tokens if word not in stopwords]
    return text
df["body_text_nonstop"]=df['body_text'].apply(lambda x: clean_text(x.lower()))
print(df.head())"""

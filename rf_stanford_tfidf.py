import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import re
import string
import numpy as np
df=pd.read_csv("datasets/stanfordSentimentTreebank/Stanford_recreated.csv")
#print(df.head(5))
ps=nltk.PorterStemmer()
#wn=nltk.WordNetLemmatizer()
stopwords=nltk.corpus.stopwords.words('english')
def clean_text(text):
    text=''.join([char for char in text if char not in string.punctuation])
    tokens=re.split('\W+',text)
    text=[ps.stem(word) for word in tokens if word not in stopwords]
    return text
df['clean_text']=df['body_text'].apply(lambda x:clean_text(x))
df=df.drop(['body_text','ID','sentiment values'],axis=1)
print(df.head(5))
df.to_csv(r'datasets/stanfordSentimentTreebank/Stanford_optimal.csv', index = False)

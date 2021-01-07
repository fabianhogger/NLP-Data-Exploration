import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import re
import string
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score

df=pd.read_csv("datasets/stanfordSentimentTreebank/Stanford_fixed2.csv")
#print(df.head(5))
ps=nltk.PorterStemmer()
#wn=nltk.WordNetLemmatizer()
stopwords=nltk.corpus.stopwords.words('english')
def clean_text(text):
    text=''.join([char for char in text if char not in string.punctuation])
    tokens=re.split('\W+',text)
    text=' '.join([ps.stem(word) for word in tokens if word not in stopwords])
    return text
df.dropna(inplace=True)
df['clean_text']=df['body_text'].apply(lambda x:clean_text(x))

df=df[df['label'].isin([1.0,0.0])]
labels=df['label'].astype('int32')
#df=df.drop(['body_text','ID','sentiment values'],axis=1)
print(df.head(5))
#df.to_csv(r'datasets/stanfordSentimentTreebank/Stanford_optimal.csv', index = False)
tfidf_vec=TfidfVectorizer(analyzer=clean_text)#Empty OBject
tfidf_fit=tfidf_vec.fit_transform(df['clean_text'])

from sklearn.model_selection import KFold,cross_val_score
rf=RandomForestClassifier(n_estimators=150,max_depth=None,n_jobs=-1)
k_fold=KFold(n_splits=5)#5 subsets of data for iterations
print(cross_val_score(rf,tfidf_fit,labels,cv=k_fold,scoring='accuracy',n_jobs=-1))
"""
cleaned body_text
n_estimators=50,max_depth=20
[0.704705   0.74093264 0.76470834 0.722589   0.73888517]
"""
"""
cleaned body_text
n_estimators=150,max_depth=None
[0.74444259 0.77339963 0.79897209 0.76888685 0.78418018]
"""

import pandas as pd
import re
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
ps=nltk.PorterStemmer()
stopwords=nltk.corpus.stopwords.words('english')
df=pd.read_csv("datasets/Usairline/tweets_clean.csv")
print(df.shape)
example=df['text']
def remove_at(text):
    txt=re.sub("@[a-zA-Z0-9]+", "", text)
    return txt
def clean_text(text):
    text=text.lower()
    text=''.join([char for char in text if char not in string.punctuation])
    tokens=re.split('\W+',text)
    text=[ps.stem(word) for word in tokens if word not in stopwords]
    return text
df['no_at']=df['text'].apply(lambda x:remove_at(x))
#df['clean_text']=df['no_at'].apply(lambda x:clean_text(x.lower()))
print(df.head())
tfidf_vec=TfidfVectorizer(analyzer=clean_text)
X_tfidf=tfidf_vec.fit_transform(df['no_at'])
from sklearn.model_selection import KFold,cross_val_score
rf=RandomForestClassifier(n_estimators=50,max_depth=20,n_jobs=-1)
k_fold=KFold(n_splits=5)#5 subsets of data for iterations
print(cross_val_score(rf,X_tfidf,df["label"],cv=k_fold,scoring='accuracy',n_jobs=-1))

"""
n_estimators=50,max_depth=20
[0.81463837 0.75519931 0.66594454 0.89514731 0.8669844 ]
"""

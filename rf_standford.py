import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
pd.set_option('display.max_colwidth',100)
df=pd.read_csv("datasets/StanfordSentimentTreeBank/Stanford_clean.csv")
print(df.head(5))

tfidf_vec=TfidfVectorizer()
X_tfidf=tfidf_vec.fit_transform(df['cleaner_text'])
print(type(X_tfidf))
print(X_tfidf.shape)

from sklearn.model_selection import KFold,cross_val_score
rf=RandomForestClassifier(n_jobs=-1)
k_fold=KFold(n_splits=5)#5 subsets of data for iterations
print(cross_val_score(rf,X_tfidf,df["label"],cv=k_fold,scoring='accuracy',n_jobs=-1))

"""
def remove(text):
    re_digits=''.join([word for word in text if  not word.isdigit()])
    return re_digits

print(df.shape)
df['cleaner_text']=df['clean_text'].apply(lambda x:remove(x))
"
df=df.drop(['clean_text'],axis=1)
df=df.loc[df['label'].isin([1,0]) ]
print(df.shape)
df.to_csv(r'datasets/StanfordSentimentTreeBank/Stanford_clean.csv', index = False)
"""

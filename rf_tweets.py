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
from sklearn.model_selection import  train_test_split

X_train,X_test,y_train,y_test=train_test_split(df[['no_at']],df['label'],test_size=0.2)


tfidf_vec=TfidfVectorizer(analyzer=clean_text)
X_tfidf_fit=tfidf_vec.fit(X_train['no_at'])

X_tfidf_train=X_tfidf_fit.transform(X_train['no_at'])
X_tfidf_test=X_tfidf_fit.transform(X_test['no_at'])

from sklearn.metrics import precision_recall_fscore_support as score
rf=RandomForestClassifier(n_estimators=150,max_depth=None,n_jobs=-1)
rf_model=rf.fit(X_tfidf_train,y_train)


print("FEATURE IMPORTANCE",sorted(zip(rf_model.feature_importances_,X_train.columns),reverse=True)[:10])
y_pred=rf_model.predict(X_tfidf_test)
precision,recall,fscore,support=score(y_test,y_pred,pos_label=1,average="binary")
print("precision={},recall={},accuracy={}".format(round(precision,3),round(recall,3),round((y_pred==y_test).sum()/len(y_test),3)))

"""from sklearn.model_selection import KFold,cross_val_score
rf=RandomForestClassifier(n_estimators=150,max_depth=None,n_jobs=-1)
k_fold=KFold(n_splits=5)#5 subsets of data for iterations
print(cross_val_score(rf,X_tfidf,df["label"],cv=k_fold,scoring='accuracy',n_jobs=-1))
"""
"""
n_estimators=50,max_depth=20
[0.81463837 0.75519931 0.66594454 0.89514731 0.8669844 ]
"""
"""
n_estimators=150,max_depth=None
[0.89735816 0.86958406 0.83102253 0.93890815 0.92634315]

"""
"""
n_estimators=50,max_depth=20
FEATURE IMPORTANCE [(0.07167417639540284, 'no_at')]
precision=0.963,recall=0.171,accuracy=0.835

"""
"""
n_estimators=150,max_depth=None

FEATURE IMPORTANCE [(0.05288006560342656, 'no_at')]
precision=0.845,recall=0.635,accuracy=0.9
"""

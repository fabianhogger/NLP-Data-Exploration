import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import re
import string
df=pd.read_csv("datasets/Combination/combination.csv")
#print(df.head(5))
ps=nltk.PorterStemmer()
stopwords=nltk.corpus.stopwords.words('english')
def clean_text(text):
    text=''.join([char for char in text if char not in string.punctuation])
    tokens=re.split('\W+',text)
    text=[ps.stem(word) for word in tokens if word not in stopwords]
    return text

from sklearn.model_selection import  train_test_split

X_train,X_test,y_train,y_test=train_test_split(df[['body_text']],df['label'],test_size=0.2)
print(X_train.head())
tfidf_vec=TfidfVectorizer(analyzer=clean_text)#Empty OBject
tfidf_fit=tfidf_vec.fit(X_train['body_text'])#Fit on train data
tfidf_train=tfidf_fit.transform(X_train['body_text'])#Transform train data using the object we fit earlier
tfidf_test=tfidf_fit.transform(X_test['body_text'])#T




tfidf_train=pd.DataFrame(tfidf_train.toarray())
X_train_vect=pd.concat([tfidf_train],axis=1)
tfidf_test=pd.DataFrame(tfidf_test.toarray())
X_test_vect=pd.concat([tfidf_test],axis=1)
print(X_train_vect.head())

from sklearn.metrics import precision_recall_fscore_support as score
rf=RandomForestClassifier(n_estimators=50,max_depth=20,n_jobs=-1)
rf_model=rf.fit(X_train_vect,y_train)


print("FEATURE IMPORTANCE",sorted(zip(rf_model.feature_importances_,X_train.columns),reverse=True)[:10])
y_pred=rf_model.predict(X_test_vect)
precision,recall,fscore,support=score(y_test,y_pred,pos_label=1,average="binary")
print("precision={},recall={},accuracy={}".format(round(precision,3),round(recall,3),round((y_pred==y_test).sum()/len(y_test),3)))
#precision=0.77,recall=0.746,accuracy=0.758

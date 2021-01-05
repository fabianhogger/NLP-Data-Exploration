import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import re
import string
df=pd.read_csv("datasets/Combination/combination.csv")
#print(df.head(5))
ps=nltk.PorterStemmer()
wn=nltk.WordNetLemmatizer()
stopwords=nltk.corpus.stopwords.words('english')
def clean_text(text):
    text=''.join([char for char in text if char not in string.punctuation])
    tokens=re.split('\W+',text)
    text=[wn.lemmatize(word) for word in tokens if word not in stopwords]
    return text
def count_punct(text):
    count=sum([1 for char in text if char in string.punctuation])
    return count/(len(text)-text.count(" "))


df['body_len']=df["body_text"].apply(lambda x:len(x)-x.count(" "))
df["punct_percent"]=df["body_text"].apply(lambda x:count_punct(x))



from sklearn.model_selection import  train_test_split

X_train,X_test,y_train,y_test=train_test_split(df[['body_text','body_len','punct_percent']],df['label'],test_size=0.2)
print(X_train.head())
tfidf_vec=TfidfVectorizer(analyzer=clean_text)#Empty OBject
tfidf_fit=tfidf_vec.fit(X_train['body_text'])#Fit on train data
tfidf_train=tfidf_fit.transform(X_train['body_text'])#Transform train data using the object we fit earlier
tfidf_test=tfidf_fit.transform(X_test['body_text'])#T




tfidf_train=pd.DataFrame(tfidf_train.toarray())
X_train_vect=pd.concat([X_train[['body_len','punct_percent']].reset_index(drop=True),tfidf_train],axis=1)
tfidf_test=pd.DataFrame(tfidf_test.toarray())
X_test_vect=pd.concat([X_test[['body_len','punct_percent']].reset_index(drop=True),tfidf_test],axis=1)
print(X_train_vect.head())

from sklearn.metrics import precision_recall_fscore_support as score
rf=RandomForestClassifier(n_estimators=150,max_depth=None,n_jobs=-1)
rf_model=rf.fit(X_train_vect,y_train)


print("FEATURE IMPORTANCE",sorted(zip(rf_model.feature_importances_,X_train.columns),reverse=True)[:10])
y_pred=rf_model.predict(X_test_vect)
precision,recall,fscore,support=score(y_test,y_pred,pos_label=1,average="binary")
print("precision={},recall={},accuracy={}".format(round(precision,3),round(recall,3),round((y_pred==y_test).sum()/len(y_test),3)))
#precision=0.77,recall=0.746,accuracy=0.758
#FEATURE IMPORTANCE [(0.01888935523262042, 'body_text'), (0.011270086616209096, 'body_len'), (0.00927560457799738, 'punct_percent')]
#precision=0.78,recall=0.709,accuracy=0.755
"""
stemming
n_estimators=150,max_depth=None
FEATURE IMPORTANCE [(0.02622180998302381, 'body_text'), (0.02189899357988389, 'body_len'), (0.011993551647256951, 'punct_percent')]
precision=0.888,recall=0.73,accuracy=0.8
"""
"""
lemmatize
n_estimators=50,max_depth=20
FEATURE IMPORTANCE [(0.014376387792706726, 'body_text'), (0.010722588357615026, 'body_len'), (0.006830777814888046, 'punct_percent')]
precision=0.807,recall=0.658,accuracy=0.735
"""
"""
lemmatize
n_estimators=150,max_depth=None
precision=0.805,recall=0.639,accuracy=0.733
"""

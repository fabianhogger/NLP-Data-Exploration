import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
pd.set_option('display.max_colwidth',100)
df=pd.read_csv("datasets/StanfordSentimentTreeBank/Stanford_clean.csv")
print(df.head(5))

from sklearn.model_selection import  train_test_split

X_train,X_test,y_train,y_test=train_test_split(df[['cleaner_text']],df['label'],test_size=0.2)
tfidf_vec=TfidfVectorizer()
X_fit=tfidf_vec.fit(X_train['cleaner_text'])
X_tf_train=X_fit.transform(X_train['cleaner_text'])
X_tf_test=X_fit.transform(X_test['cleaner_text'])

from sklearn.metrics import precision_recall_fscore_support as score
rf=RandomForestClassifier(n_estimators=100,max_depth=20,n_jobs=-1)
rf_model=rf.fit(X_tf_train,y_train)

print("FEATURE IMPORTANCE",sorted(zip(rf_model.feature_importances_,X_train.columns),reverse=True)[:10])
y_pred=rf_model.predict(X_tf_test)
precision,recall,fscore,support=score(y_test,y_pred,pos_label=1,average="binary")
print("precision={},recall={},accuracy={}".format(round(precision,3),round(recall,3),round((y_pred==y_test).sum()/len(y_test),3)))

"""from sklearn.model_selection import KFold,cross_val_score
rf=RandomForestClassifier(n_estimators=150,max_depth=None,n_jobs=-1)
k_fold=KFold(n_splits=5)#5 subsets of data for iterations
print(cross_val_score(rf,X_tfidf,df["label"],cv=k_fold,scoring='accuracy',n_jobs=-1))
"""
"""
cross_val_score
stemming
    n_estimators 100,max_depth=None
[0.89727428 0.89168093 0.88717285 0.8699282  0.88549841]
"""
"""
cross_val_score
stemming
    n_estimators 150,max_depth=None
[0.89589682 0.89151396 0.88754852 0.86967774 0.88591585]
"""
"""

n_estimators=150,max_depth=None
FEATURE IMPORTANCE [(2.4852766022297162e-08, 'cleaner_text')]
precision=0.897,recall=0.929,accuracy=0.902
"""

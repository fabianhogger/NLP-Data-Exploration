import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
pd.set_option('display.max_colwidth',100)
df=pd.read_csv("datasets/StanfordSentimentTreeBank/Stanford_clean.csv")
print(df.head(5))

from sklearn.model_selection import  train_test_split
from sklearn.metrics import precision_recall_fscore_support as score

#X_train,X_test,y_train,y_test=train_test_split(df[['cleaner_text']],df['label'],test_size=0.2)
count_vect=CountVectorizer()
X_count=count_vect.fit_transform(df['cleaner_text'])
#X_count_train=X_count.transform(X_train['cleaner_text'])
#X_count_test=X_count.transform(X_test['cleaner_text'])

"""rf=RandomForestClassifier(n_estimators=150,max_depth=None,n_jobs=-1)
rf_model=rf.fit(X_count_train,y_train)


print("FEATURE IMPORTANCE",sorted(zip(rf_model.feature_importances_,X_train.columns),reverse=True)[:10])
y_pred=rf_model.predict(X_count_test)
precision,recall,fscore,support=score(y_test,y_pred,pos_label=1,average="binary")
print("precision={},recall={},accuracy={}".format(round(precision,3),round(recall,3),round((y_pred==y_test).sum()/len(y_test),3)))
"""

from sklearn.model_selection import KFold,cross_val_score
rf=RandomForestClassifier(n_estimators=150,max_depth=None,n_jobs=-1)
k_fold=KFold(n_splits=5)#5 subsets of data for iterations
print(cross_val_score(rf,X_count,df["label"],cv=k_fold,scoring='accuracy',n_jobs=-1))

"""
n_estimators=50,max_depth=20
precision=0.613,recall=0.98,accuracy=0.65
"""
"""
n_estimators=150,max_depth=None
precision=0.898,recall=0.943,accuracy=0.909
"""
"""
n_estimators=50,max_depth=20
[0.67354009 0.66702843 0.65116667 0.58286024 0.60181165]
"""
"""
n_estimators=150,max_depth=None
[0.89973703 0.89113829 0.88988605 0.87481216 0.88650025]
"""

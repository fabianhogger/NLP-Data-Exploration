import pandas as pd
import nltk
import re
import string
from sklearn.ensemble import RandomForestClassifier


df=pd.read_csv("datasets/Combination/combination.csv")
print(df.head(5))
ps=nltk.PorterStemmer()
stopwords=nltk.corpus.stopwords.words('english')
def clean_text(text):
    text=''.join([char for char in text if char not in string.punctuation])
    tokens=re.split('\W+',text)
    text=[ps.stem(word) for word in tokens if word not in stopwords]
    return text

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import  train_test_split

X_train,X_test,y_train,y_test=train_test_split(df[['body_text']],df['label'],test_size=0.2)
count_vect=CountVectorizer(analyzer=clean_text)
X_counts_fit=count_vect.fit(X_train['body_text'])
X_train_count=X_counts_fit.transform(X_train['body_text'])
X_test_count=X_counts_fit.transform(X_test['body_text'])

X_train_count=pd.DataFrame(X_train_count.toarray())
X_test_count=pd.DataFrame(X_test_count.toarray())
from sklearn.metrics import precision_recall_fscore_support as score
rf=RandomForestClassifier(n_estimators=150,max_depth=None,n_jobs=-1)
rf_model=rf.fit(X_train_count,y_train)

print("FEATURE IMPORTANCE",sorted(zip(rf_model.feature_importances_,X_train.columns),reverse=True)[:10])
y_pred=rf_model.predict(X_test_count)
precision,recall,fscore,support=score(y_test,y_pred,pos_label=1,average="binary")
print("precision={},recall={},accuracy={}".format(round(precision,3),round(recall,3),round((y_pred==y_test).sum()/len(y_test),3)))
"""
FEATURE IMPORTANCE [(0.06695298628623501, 1728), (0.043137360198200365, 418), (0.03893031817218898, 2274), (0.030679850095938526, 1698), (0.026339039536004805, 1387), (0.021229975155907393, 4114), (0.0190946334228831, 2859), (0.01705063820820016, 4225), (0.017032145694018275, 2583), (0.016900318166265464, 2903)]
precision=0.73,recall=0.833,accuracy=0.773
"""
"""
FEATURE IMPORTANCE [(0.047104468261371366, 1728), (0.021415349481218537, 1698), (0.0204382582829047, 2274), (0.015495684802355155, 418), (0.01369200651937918, 1387), (0.010846805822180897, 1144), (0.009084474057207375, 2550), (0.00858436744765013, 115), (0.007887952074728045, 275), (0.007474314252670187, 401)]
precision=0.845,recall=0.707,accuracy=0.793
"""
"""
FEATURE IMPORTANCE [(0.006252591103621767, 'body_text')]
precision=0.798,recall=0.792,accuracy=0.804
"""

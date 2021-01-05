import pandas as pd
import string
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support as score

def clean_text(text):
    text=''.join([char.lower() for char in text if char not in string.punctuation])
    tokens=re.split('\W+',text)
    text=' '.join([ps.stem(word) for word in tokens if word not in stopwords])#returns a string for n-gram
    return text

pd.set_option('display.max_colwidth',100)
df=pd.read_csv("datasets/Combination/combination.csv")

stopwords=nltk.corpus.stopwords.words('english')
ps=nltk.PorterStemmer()

df["clean_text"]=df['body_text'].apply(lambda x: clean_text(x))


print(df.head())
from sklearn.model_selection import  train_test_split

X_train,X_test,y_train,y_test=train_test_split(df[['clean_text']],df['label'],test_size=0.2)

ngram_vect=CountVectorizer(ngram_range=(2,2))
print(ngram_vect)

X_counts_fit=ngram_vect.fit(X_train['clean_text'])
X_counts=X_counts_fit.transform(X_train['clean_text'])

X_counts_test=X_counts_fit.transform(X_test['clean_text'])

print(X_counts.shape)

#print(ngram_vect.get_feature_names())
rf=RandomForestClassifier(n_estimators=150,max_depth=None,n_jobs=-1)
rf_model=rf.fit(X_counts,y_train)
y_pred=rf_model.predict(X_counts_test)
precision,recall,fscore,support=score(y_test,y_pred,pos_label=1,average="binary")
print("precision={},recall={},accuracy={}".format(round(precision,3),round(recall,3),round((y_pred==y_test).sum()/len(y_test),3)))


"""
ngram_range=(2,2)
n_estimators=150,max_depth=None
precision=0.851,recall=0.289,accuracy=0.616
"""

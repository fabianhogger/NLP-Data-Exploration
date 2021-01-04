import pandas as pd

def create_label(sent_val):
    if sent_val>=0.5:
        return 1
    else:
        return 0
df=pd.read_csv("datasets/stanfordSentimentTreebank/dictionary.txt",sep='|')
df2=pd.read_csv("datasets/stanfordSentimentTreebank/sentiment_labels.txt",sep='|')
df.columns=['body_text','phrase ids']

print(df.head(5))
#print(df['phrase ids'].value_counts)
#print(df2['phrase ids'].value_counts)
df3=df.merge(df2,how='left',on='phrase ids')
print(df3.head(5))
df3=df3.rename(columns={"phrase ids":"ID"})


print(df3.head(5))
print(df3['sentiment values'].max())
print(df3['sentiment values'].min())
df3['label']=df3['sentiment values'].apply(lambda x:create_label(x))
print(df3.head(5))

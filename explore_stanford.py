import pandas as pd


df=pd.read_csv("datasets/stanfordSentimentTreebank/dictionary.txt",sep='|')
df2=pd.read_csv("datasets/stanfordSentimentTreebank/sentiment_labels.txt",sep='|')
df.columns=['body_text','phrase ids']

print(df.head(5))
#print(df['phrase ids'].value_counts)
#print(df2['phrase ids'].value_counts)
df3=df.merge(df2,how='left',on='phrase ids')
print(df3.head(5))
df3=df3.rename(columns={"phrase ids":"ID","sentiment values":"label"})
print(df3.head(5))

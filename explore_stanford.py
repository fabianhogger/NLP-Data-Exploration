import pandas as pd


df=pd.read_csv("datasets/stanfordSentimentTreebank/datasetSentences.txt",sep='\t')
print(df.head(5))

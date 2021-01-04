import pandas as pd
#Exploring airline tweets
df=pd.read_csv("datasets/Tweets.csv")
print(df.head(5))
print(df.shape)
#shape  14640 rows 15 columns
print(df[df['airline_sentiment']=='neutral'].count())
#negative 9178
#positive 2363
#neutral 3099

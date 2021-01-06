import pandas as pd
import matplotlib.pyplot as plt
#Exploring airline tweets
df=pd.read_csv("datasets/Usairline/Tweets.csv")
#print(df.head(5))
#print(df.shape)
#shape  14640 rows 15 columns
#print(df[df['airline_sentiment']=='neutral'].count())
#negative 9178
#positive 2363
#neutral 3099
#random row
"""print(df['text'][30],df['airline_sentiment'][30])
print(df['tweet_coord'][30])
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'positive','negative','neutral'
sizes = [2363,9178,3099]
explode = (0,0,0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()"""
def make_label(word):
    if word== "positive":
        return 1
    elif word== "neutral":
        return 0.5
    elif word== "negative":
        return 0
df=df[['text','airline_sentiment']]
df['label']=df['airline_sentiment'].apply(lambda x:make_label(x))
print(df.head(5))

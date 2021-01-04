import pandas as pd
import matplotlib.pyplot as plt
def create_label(sent_val):
    if sent_val>=0.6:
        return 1
    elif sent_val<0.6 and sent_val>=0.4:
        return 0.5
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
print("Dataframe shape",df3.shape)
#(239231, 4)
print("Number of samples",df3.label.value_counts())
""" for 2 labels
 1    151412
0     87819
"""
""" for 3 labels
0.5    119448
1.0     65403
0.0     54380
"""

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'positive','neutral','negative'
sizes = [65403,119448,54380]
explode = (0,0,0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
df3.to_csv(r'datasets/stanfordSentimentTreebank/Stanford_recreated.csv', index = False)

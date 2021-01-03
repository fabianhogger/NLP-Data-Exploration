import pandas as pd
import string
import re
import nltk
import glob, os
import matplotlib.pyplot as plt
stopwords=nltk.corpus.stopwords.words('english')
all_files = glob.glob(os.path.join('datasets/Combination', "*.txt"))     # advisable to use os.path.join as this makes concatenation OS independent

df_from_each_file = (pd.read_csv(f,sep='\t',header=None) for f in all_files)
concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)
concatenated_df.columns=['body_text','label']
#print(concatenated_df.head())

concatenated_df.to_csv(r'datasets/Combination/multimedia.csv', index = False)

def clean_text(text):
    text=''.join([char for char in text if char not in string.punctuation])
    tokens=re.split('\W+',text)
    text=[word for word in tokens if word not in stopwords]
    return text
concatenated_df["body_text_nonstop"]=concatenated_df['body_text'].apply(lambda x: clean_text(x.lower()))
print(concatenated_df.head())
print("Dataframe shape",concatenated_df.shape)
print("Number of samples",concatenated_df.label.value_counts())
"""
Number of samples 1    1386
0    1362
Name: label, dtype: int64
"""
stemmed=list(concatenated_df['body_text_nonstop'])
counts=[len(row) for row in stemmed]
print("Average number of tokens for each sentence after stemming",sum(counts)/len(counts))
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'positive','negative'
sizes = [1385,1362]
explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
#Dataframe shape 2748, 3

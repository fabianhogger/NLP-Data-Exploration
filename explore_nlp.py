import pandas as pd
import string
import re
import nltk
import glob, os
stopwords=nltk.corpus.stopwords.words('english')
all_files = glob.glob(os.path.join('datasets/Combination', "*.txt"))     # advisable to use os.path.join as this makes concatenation OS independent

df_from_each_file = (pd.read_csv(f,sep='\t',header=None) for f in all_files)
concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)
concatenated_df.columns=['body_text','label']
print(concatenated_df.head())

concatenated_df.to_csv(r'datasets/Combination/multimedia.csv', index = False)

"""
all_files=[]
for dirname, _, filenames in os.walk('datasets/Combination'):
    for filename in filenames:
        all_files.append(filename)
#print(all_files)
all_df=[]
for file in all_files:
    df=pd.read_csv(os.path.join('datasets/Combination',file),sep='\t',header=None)
    all_df.append(df)

for i in range(0,len(all_df)):
    if i>0:
        df=pd.concat(all_df[i-1],all_df[i])


df=pd.read_csv("datasets/Combination/amazon_cells_labelled.txt",sep='\t',header=None)
df.columns=['body_text','label']
print(df.head())
def clean_text(text):
    text=''.join([char for char in text if char not in string.punctuation])
    tokens=re.split('\W+',text)
    text=[word for word in tokens if word not in stopwords]
    return text
df["body_text_nonstop"]=df['body_text'].apply(lambda x: clean_text(x.lower()))
print(df.head())
"""

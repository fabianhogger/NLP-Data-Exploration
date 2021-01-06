import pandas as pd
import string
df=pd.read_csv("datasets/StanfordSentimentTreeBank/Stanford_optimal.csv")
print(df.head(5))

def remove(text):
    re_digits=''.join([word for word in text if  not word.isdigit()])
    return re_digits

print(df.shape)
df['cleaner_text']=df['clean_text'].apply(lambda x:remove(x))

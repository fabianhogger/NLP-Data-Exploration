import nltk
import re
import string
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
ps=nltk.PorterStemmer()
stopwords=nltk.corpus.stopwords.words('english')
#Load Tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

#tokenizer=Tokenizer()
def clean_text(text):
    text=''.join([char for char in text if char not in string.punctuation and not char.isdigit()])
    tokens=re.split('\W+',text)
    text=' '.join([ps.stem(word) for word in tokens if word not in stopwords])
    return text


def process_for_sentiment(sentences):
    sentences=[clean_text(sentence) for sentence in sentences]
    sequences=tokenizer.texts_to_sequences(sentences)#replaces the words with their indexes
    #Padding
    sequences_padded=pad_sequences(sequences,50)
    return sequences_padded



def get_sentiment(sentences):
    model=keras.models.load_model("SentiModel")
    sequences=process_for_sentiment(sentences)
    print(sequences)
    predictions=model.predict(sequences)
    print(predictions)
    classes = (predictions>0.25)
    print(classes)
    #calculate final prediction
    #final_prediction=
    #return final_prediction
#Initialize and fit the tokenizer
sent=['The officials said they had to allow for the possibility that Bush would make some change in the plan, but a source close to the White House said it was “all but set in stone”, the Post reported.', 'Ending the tariffs 16 months before schedule could spark a political backlash against Bush in next year’s presidential election in the pivotal steel-producing states of Ohio, Pennsylvania and West Virginia.', 'The Washington Post sources said Bush’s aides concluded they could not run the risk that the European Union would carry out its threat to impose sanctions on citrus fruit from Florida, farm machinery, textiles and other products.', 'The Bush administration imposed the duties, initially for up to 30%, in 2002 to help defend the country’s struggling steel industry against cheap imports', 'Quoting administration and industry sources, The Washington Post said in its Monday editions that President George Bush is\xa0 likely to announce the decision this week.', 'The Bush administration has decided to repeal its 20-month-old tariffs on imported steel to head off a trade war that would have included foreign retaliation against products from politically crucial US states.']
get_sentiment(sent)
"""
{'Bush': ['The officials said they had to allow for the possibility that Bush would make some change in the plan, but a source close to the White House said it was “all but set in stone”, the Post reported.', 'Ending the tariffs 16 months before schedule could spark a political backlash against Bush in next year’s presidential election in the pivotal steel-producing states of Ohio, Pennsylvania and West Virginia.', 'The Washington Post sources said Bush’s aides concluded they could not run the risk that the European Union would carry out its threat to impose sanctions on citrus fruit from Florida, farm machinery, textiles and other products.', 'The Bush administration imposed the duties, initially for up to 30%, in 2002 to help defend the country’s struggling steel industry against cheap imports', 'Quoting administration and industry sources, The Washington Post said in its Monday editions that President George Bush is\xa0 likely to announce the decision this week.', 'The Bush administration has decided to repeal its 20-month-old tariffs on imported steel to head off a trade war that would have included foreign retaliation against products from politically crucial US states.'], 'White House': ['A source involved in the negotiations said White House aides looked for some step short of a full repeal that would satisfy the European Union, but concluded that it was “technically possible but practically impossible”, according to the Post.', 'The officials said they had to allow for the possibility that Bush would make some change in the plan, but a source close to the White House said it was “all but set in stone”, the Post reported.', 'A spokesman for the White House denied a decision had been made to repeal the tariffs.'], 'Post': ['A source involved in the negotiations said White House aides looked for some step short of a full repeal that would satisfy the European Union, but concluded that it was “technically possible but practically impossible”, according to the Post.', 'The officials said they had to allow for the possibility that Bush would make some change in the plan, but a source close to the White House said it was “all but set in stone”, the Post reported.', 'The Washington Post sources said Bush’s aides concluded they could not run the risk that the European Union would carry out its threat to impose sanctions on citrus fruit from Florida, farm machinery, textiles and other products.', 'Quoting administration and industry sources, The Washington Post said in its Monday editions that President George Bush is\xa0 likely to announce the decision this week.'], 'West Virginia': ['Ending the tariffs 16 months before schedule could spark a political backlash against Bush in next year’s presidential election in the pivotal steel-producing states of Ohio, Pennsylvania and West Virginia.'], 'Pennsylvania': ['Ending the tariffs 16 months before schedule could spark a political backlash against Bush in next year’s presidential election in the pivotal steel-producing states of Ohio, Pennsylvania and West Virginia.'], 'The European Union': ['A source involved in the negotiations said White House aides looked for some step short of a full repeal that would satisfy the European Union, but concluded that it was “technically possible but practically impossible”, according to the Post.', 'The European Union, one of a number of trade partners to take action at the WTO over the levies, had warned it was ready to hit Washington with sanctions on up to $2.2 billion of goods within five days of the WTO approving the court ruling.', 'The Washington Post sources said Bush’s aides concluded they could not run the risk that the European Union would carry out its threat to impose sanctions on citrus fruit from Florida, farm machinery, textiles and other products.'], 'Florida': ['The Washington Post sources said Bush’s aides concluded they could not run the risk that the European Union would carry out its threat to impose sanctions on citrus fruit from Florida, farm machinery, textiles and other products.'], 'Ohio': ['Ending the tariffs 16 months before schedule could spark a political backlash against Bush in next year’s presidential election in the pivotal steel-producing states of Ohio, Pennsylvania and West Virginia.'], 'Washington': ['The European Union, one of a number of trade partners to take action at the WTO over the levies, had warned it was ready to hit Washington with sanctions on up to $2.2 billion of goods within five days of the WTO approving the court ruling.', 'The Washington Post sources said Bush’s aides concluded they could not run the risk that the European Union would carry out its threat to impose sanctions on citrus fruit from Florida, farm machinery, textiles and other products.', 'Speculation had mounted that Washington would scrap or roll back the controversial tariffs after it last week sought and obtained an effective delay in retaliatory sanctions by countries opposed to them.', 'Quoting administration and industry sources, The Washington Post said in its Monday editions that President George Bush is\xa0 likely to announce the decision this week.'], 'WTO': ['The European Union, one of a number of trade partners to take action at the WTO over the levies, had warned it was ready to hit Washington with sanctions on up to $2.2 billion of goods within five days of the WTO approving the court ruling.'], 'US': ['The Bush administration has decided to repeal its 20-month-old tariffs on imported steel to head off a trade war that would have included foreign retaliation against products from politically crucial US states.']}

"""

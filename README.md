# NLP-Data-Exploration

Data exploration for some sentiment analysis datasets

## Combination of reviews

The first dataset is a small one of **2748** reviews labelled as negative or positive.
The dataset is a combination of amazon ,imdb and yelp reviews.

![pie chart](/datasets/Combination/combination_pie_chart.png "Label distribution")

Using this code we created a unified file to work on later: [exploration_nlp](https://github.com/fabianhoegger/NLP-Data-Exploration/exploration_nlp.py)
The csv file is available here:[multimedia.csv](https://github.com/fabianhoegger/NLP-Data-Exploration/tree/main/datasets/Combination)

## Stanford Dataset

The Stanford dataset is much bigger with **239231** total movie reviews from rotten tomatoes.
The dataset comes with annotated reviews in a score of 0 to 1 meaning:
0-0.2 very negative  0.2-0.4 negative 0.4-0.6 neutral and 0.6-0.8 positive ,0.8-0.1 very positive.
Bellow is a simplified distribution chart

![pie chart 2](/datasets/stanfordSentimentTreebank/standford_pie_neutral.png "Label distribution")

Using this code we created a unified file to work on later: [exploration_stanford](https://github.com/fabianhoegger/NLP-Data-Exploration/exploration_stanford.py)

## US Airline Tweet Dataset

The US Airline tweet dataset is made of **14640** tweets each annotated as positive,neutral or negative .Additionally it has a "negativereason" where it states the reason that a tweet is negative and even a negativereason confidence level column.
Bellow there's a chart showing the percentage of positive,negative and neutral tweets

![tweet chart ](/datasets/USairline/pie_chart.png "Label distribution")

Data exploration code for this dataset: [explore_tweets](https://github.com/fabianhoegger/NLP-Data-Exploration/explore_tweets.py)


## Random Forest Classification

Using the RandomForestClassifier provided by the sklearn library we tried to train the algorithm with the data preprocessed in different ways and we made some changes in the algorithm's parameters to see how it affected the result.


### Combination of reviews


| Method | n_estimators=50,max_depth=20| n_estimators=50,max_depth=20 |
| Stemming | :---: | :---: | :---: | :---: || :---: | :---: |
| Lemmatizing | 301 | 283 |

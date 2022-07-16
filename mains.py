import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt  #for plotting our figure
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import wordcloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB


data = pd.read_csv("stress.csv")
print(data.head())

#checking if contains null values or not
print(data.isnull().sum())


nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
stopword=set(stopwords.words('english'))


def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text


data["text"] = data["text"].apply(clean)

#looking at most use texts by visualising a word cloud

text = " ".join(i for i in data.text)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

data["label"] = data["label"].map({0: "No Stress", 1: "stress"})
data = data[["text", "label"]]
print(data.head())


#split dataset into training and test sets.

x = np.array(data["text"])
y = np.array(data["label"])

cv = CountVectorizer()
X = cv.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33, random_state=42)

model = BernoulliNB()
model.fit(xtrain, ytrain)

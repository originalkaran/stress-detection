import pandas as pd
import numpy as np
data = pd.read_csv("stress.csv")
print(data.head())

#checking if contains null values or not
print(data.isnull().sum())

import nltk
import re
nltk.downlaod('stopwords')
stemmer = nltk.SnowballStemer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
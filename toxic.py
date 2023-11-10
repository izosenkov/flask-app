from pymystem3 import Mystem
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from deps import load_models
import pandas as pd
from nltk.tokenize import word_tokenize

vectorizer, lr = load_models()

nltk.download('stopwords')

stop_words = set(stopwords.words('russian'))


snowball = SnowballStemmer(language="russian")
lemmatizer = Mystem()

def preprocess(text):
    text = list(filter(str.isalpha, word_tokenize(text.lower())))
    text = lemmatizer.lemmatize(' '.join(text))[::2]
    text = list(word for word in text if word not in stop_words)
    text = [snowball.stem(word) for word in text]
    return ' '.join(text)

def get_score(text):
    return lr.predict_proba(vectorizer.transform(pd.Series(preprocess(text.replace('\n', ' ')))))[0][1]


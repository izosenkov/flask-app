import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

def load_models():
    with open('models/vectorizer2.pickle', 'rb') as fp:
        vectorizer = pickle.load(fp)

    with open('models/lr.pickle', 'rb') as fp:
        lr = pickle.load(fp)

    return vectorizer, lr

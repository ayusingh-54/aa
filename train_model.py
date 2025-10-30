import os
import string
import pandas as pd
import numpy as np
import joblib
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_data(fake_path='data/Fake.csv', true_path='data/True.csv'):
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)
    fake['target'] = 'fake'
    true['target'] = 'true'
    data = pd.concat([fake, true]).reset_index(drop=True)
    return data


def preprocess_text(series):
    # lowercase
    series = series.str.lower()

    # remove punctuation
    table = str.maketrans('', '', string.punctuation)
    series = series.apply(lambda x: x.translate(table))

    # remove stopwords
    try:
        stop = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        stop = set(stopwords.words('english'))

    series = series.apply(lambda x: ' '.join([w for w in x.split() if w not in stop]))
    return series


def train_and_save(model_path='models/news_classifier.pkl'):
    print('Loading data...')
    data = load_data()

    # Drop columns not needed if present (safe guards similar to notebook)
    for col in ['date', 'title']:
        if col in data.columns:
            data.drop([col], axis=1, inplace=True)

    print('Preprocessing text...')
    data['text'] = preprocess_text(data['text'].astype(str))

    X = data['text']
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('Building pipeline...')
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('model', LogisticRegression(solver='liblinear', random_state=42))
    ])

    print('Training model...')
    pipeline.fit(X_train, y_train)

    print('Evaluating...')
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f'Validation accuracy: {acc*100:.2f}%')

    # ensure models dir exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f'Model saved to: {model_path}')


if __name__ == '__main__':
    train_and_save()

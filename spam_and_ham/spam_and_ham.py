#!./../env/bin/python3

from __future__ import annotations

import string
from collections import Counter
from os import path

import nltk
import csv
import pickle
import sklearn
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from maalepinde.maalepinde_parser import hent_maalepinde, EXCEL_FILE, remove_maalepind_clutter
from maalepinde import maalepinde_parser
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score, learning_curve


SPAM_CSV: str = 'dataset/spam.csv'


def _spamham_wordcloud(data, show: bool = True):

    ham_words = ''
    spam_words = ''

    # Creating a corpus of spam messages
    for val in data[data['label'] == 'spam'].text:
        text = val.lower()
        tokens = nltk.word_tokenize(text)
        for words in tokens:
            spam_words = spam_words + words + ' '

    # Creating a corpus of ham messages
    for val in data[data['label'] == 'ham'].text:
        text = val.lower()
        tokens = nltk.word_tokenize(text)
        for words in tokens:
            ham_words = ham_words + words + ' '

    spam_wordcloud = WordCloud(width=500, height=300).generate(spam_words)
    ham_wordcloud = WordCloud(width=500, height=300).generate(ham_words)

    if show is True:

        # Spam Word cloud
        plt.figure(figsize=(10, 8), facecolor='w')
        plt.imshow(spam_wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()

        # Creating Ham wordcloud
        plt.figure(figsize=(10, 8), facecolor='g')
        plt.imshow(ham_wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()

    return (spam_wordcloud, ham_wordcloud)


def main(gen_wordcloud: bool = True, verbose: bool = True) -> None:

    data = pd.read_csv(SPAM_CSV, encoding='latin-1')
    # data.head()

    data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
    data = data.rename(columns={"v2": "text", "v1": "label"})
    print(data[1990:2000])

    data['label'].value_counts()

    nltk.download("punkt")
    warnings.filterwarnings('ignore')

    if gen_wordcloud is True:
        _spamham_wordcloud(data)

    data = data.replace(['ham', 'spam'], [0, 1])
    # data.head(10)

    # We expect to find the Excel file in the same directory as the maalepinde script.
    excel_file: str = path.join(path.dirname(maalepinde_parser.__file__), EXCEL_FILE)
    maalepinde: DataFrame = hent_maalepinde(excel_file)
    maalepinde: DataFrame = remove_maalepind_clutter(maalepinde)

    maalepinde: DataFrame = maalepinde["MÅLPINDE"]
    maalepinde['v1'] = 3
    data = pd.concat([data, maalepinde])

    nltk.download('stopwords')

    def text_process(text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = [word for word in text.split() if word.lower() not in stopwords.words('english')]

        return " ".join(text)

    data['text'] = data['text'].apply(text_process)
    data.head()

    text = pd.DataFrame(data['text'])

    ## Counting how many times a word appears in the dataset
    total_counts = Counter()
    for i in range(len(text)):
        for word in text.values[i][0].split(" "):
            total_counts[word] += 1

    if verbose is True:
        print("Total words in data set: ", len(total_counts))

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(data['text'])
    if verbose:
        print(vectors.shape)

    features = vectors

    # split the dataset into train and test set
    X_train, X_test, y_train, y_test = train_test_split(features, data['label'], test_size=0.15, random_state=111)

    # initialize Naive Bayes classification models
    mnb = MultinomialNB(alpha=0.2)

    # fit the data onto the models
    def train(clf, features, targets):
        clf.fit(features, targets)

    train(mnb, X_train, y_train)

    newtext = ["Free entry"]
    integers = vectorizer.transform(newtext)

    x = mnb.predict(integers)

    # Naive Bayes
    y_pred_nb = mnb.predict(X_test)
    y_true_nb = y_test
    cm = confusion_matrix(y_true_nb, y_pred_nb)
    f, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax)
    plt.xlabel("y_pred_nb")
    plt.ylabel("y_true_nb")
    plt.show()


if __name__ == '__main__':
    main(gen_wordcloud=False)

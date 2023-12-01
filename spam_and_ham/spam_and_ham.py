#!./../env/bin/python3

from __future__ import annotations

import string
from collections import Counter

import nltk
import csv
import pickle
import sklearn
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score, learning_curve


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
    data = pd.read_csv('dataset/spam.csv', encoding='latin-1')
    data.head()

    data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
    data = data.rename(columns={"v2": "text", "v1": "label"})
    print(data[1990:2000])

    data['label'].value_counts()

    nltk.download("punkt")
    warnings.filterwarnings('ignore')

    if gen_wordcloud is True:
        _spamham_wordcloud(data)

    data = data.replace(['ham', 'spam'], [0, 1])
    data.head(10)

    nltk.download('stopwords')

    def text_process(text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = [word for word in text.split() if word.lower() not in stopwords.words('english')]

        return " ".join(text)

    data['text'] = data['text'].apply(text_process)
    data.head()

    text = pd.DataFrame(data['text'])
    label = pd.DataFrame(data['label'])

    ## Counting how many times a word appears in the dataset
    total_counts = Counter()
    for i in range(len(text)):
        for word in text.values[i][0].split(" "):
            total_counts[word] += 1

    if verbose is True:
        print("Total words in data set: ", len(total_counts))

    # Sorting in decreasing order (Word with the highest frequency appears first)
    vocab = sorted(total_counts, key=total_counts.get, reverse=True)

    if verbose is True:
        print(vocab[:60])

    # Mapping from words to index
    vocab_size = len(vocab)
    word2idx = {}
    #print vocab_size
    for i, word in enumerate(vocab):
        word2idx[word] = i

    # Text to Vector
    def text_to_vector(text):
        word_vector = np.zeros(vocab_size)
        for word in text.split(" "):
            if word2idx.get(word) is None:
                continue
            else:
                word_vector[word2idx.get(word)] += 1
        return np.array(word_vector)

    # Convert all titles to vectors
    word_vectors = np.zeros((len(text), len(vocab)), dtype=np.int_)
    for i, (_, text_) in enumerate(text.iterrows()):
        word_vectors[i] = text_to_vector(text_[0])

    if verbose:
        print(word_vectors.shape)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(data['text'])
    if verbose:
        print(vectors.shape)

    features = vectors

    # split the dataset into train and test set
    X_train, X_test, y_train, y_test = train_test_split(features, data['label'], test_size=0.15, random_state=111)

    # initialize multiple classification models
    svc = SVC(kernel='sigmoid', gamma=1.0)
    knc = KNeighborsClassifier(n_neighbors=49)
    mnb = MultinomialNB(alpha=0.2)
    dtc = DecisionTreeClassifier(min_samples_split=7, random_state=111)
    lrc = LogisticRegression(solver='liblinear', penalty='l1')
    rfc = RandomForestClassifier(n_estimators=31, random_state=111)

    # create a dictionary of variables and models
    clfs = {'SVC': svc, 'KN': knc, 'NB': mnb, 'DT': dtc, 'LR': lrc, 'RF': rfc}

    # fit the data onto the models
    def train(clf, features, targets):
        clf.fit(features, targets)

    def predict(clf, features):
        return (clf.predict(features))

    pred_scores_word_vectors = []
    for k, v in clfs.items():
        train(v, X_train, y_train)
        pred = predict(v, X_test)
        pred_scores_word_vectors.append((k, [accuracy_score(y_test, pred)]))

    if verbose:
        print(pred_scores_word_vectors)

    # write functions to detect if the message is spam or not
    def find(x):
        if x == 1:
            print("Message is SPAM")
        else:
            print("Message is NOT Spam")

    newtext = ["Free entry"]
    integers = vectorizer.transform(newtext)

    x = mnb.predict(integers)
    find(x)

    # Naive Bayes
    y_pred_nb = svc.predict(X_test)
    y_true_nb = y_test
    cm = confusion_matrix(y_true_nb, y_pred_nb)
    f, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax)
    plt.xlabel("y_pred_nb")
    plt.ylabel("y_true_nb")
    plt.show()


if __name__ == '__main__':
    main(gen_wordcloud=False)

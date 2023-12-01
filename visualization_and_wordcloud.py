#!env/bin/python3

import nltk
import matplotlib
import numpy as np
from string import punctuation
from wordcloud import WordCloud
from nltk.corpus import stopwords
from utils import MAALEPINDE_FORMAT
from matplotlib import pyplot as plt


def remove_stopwords(maalepind_list: MAALEPINDE_FORMAT):
    """
    Remove stopwords and punctuation

    :param maalepind_list:
    :return:
    """

    nltk.download('stopwords')

    maalpind_removed_stopwords = []
    # For fast lookups of stopwords
    stopword_set = set(stopwords.words('danish'))

    for fagnr, maalepind_set in maalepind_list.items():
        for maalepind_tpl in maalepind_set:
            for maalepind_str in maalepind_tpl[1]:
                # Remove punctuation
                maalepind_no_punct = ''.join(char for char in maalepind_str if char not in punctuation)

                # Remove stopwords
                words_without_stopwords = [word for word in maalepind_no_punct.split() if word.lower() not in stopword_set]

                # Join the words back into a sentence
                maalepind_no_stopwords = ' '.join(words_without_stopwords)

                maalpind_removed_stopwords.append(maalepind_no_stopwords)

    return maalpind_removed_stopwords


def draw_diagram_words_used(maalepinde: list[str]):
    """
    Draw the diagram using the processed data

    :param data:
    :return:
    """

    # Extract words and their occurrences for plotting
    words_occurrences = {}
    for maalepind_no_stopwords in maalepinde:
        words = maalepind_no_stopwords.split()
        for word in words:
            if word in words_occurrences:
                words_occurrences[word] += 1
            else:
                words_occurrences[word] = 1

    # Get the top 10 words and their occurrences
    top_words = sorted(words_occurrences.items(), key=lambda x: x[1], reverse=True)[:10]
    words, occurrences = zip(*top_words)

    font1 = {'family': 'serif', 'color': 'black', 'size': 30}
    font2 = {'family': 'serif', 'color': 'green', 'size': 15}

    plt.barh(words, occurrences, color="darkgreen")
    plt.title("Word Occurrence", fontdict=font1)
    plt.xlabel("Occurrence", fontdict=font2)
    plt.ylabel("Words", fontdict=font2)
    plt.show()


def word_cloud(data: list[str]) -> None:
    meta_text: str = ' '.join(data)
    dc: dict = {}
    for word in meta_text.split():
        if word not in dc:
            dc[word] = 0
        dc[word] += 1
    wc = WordCloud(background_color='black', width=800, height=500).generate_from_frequencies(dc)
    plt.axis("off")

    plt.imshow(wc)
    plt.show()

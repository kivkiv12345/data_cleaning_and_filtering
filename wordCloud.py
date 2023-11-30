import nltk
from nltk.corpus import stopwords


def word_cloud():
    # Remove specific words such as "og, i, jeg, det, at , en
    nltk.download('stopwords')

    print(stopwords.words("danish"))


if __name__ == '__main__':
    word_cloud()

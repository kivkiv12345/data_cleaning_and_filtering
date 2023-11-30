import nltk
from nltk.corpus import stopwords


def word_cloud():
    # Remove specific words such as "og, i, jeg, det, at , en
    nltk.download('stopwords')

    print(stopwords.words("danish"))
    # Remove coma and full stop
    # Make a list of these words


if __name__ == '__main__':
    word_cloud()

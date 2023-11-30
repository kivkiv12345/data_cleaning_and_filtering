import nltk
from nltk.corpus import stopwords

if __name__ == '__main__':
    # Remove specific words such as "og, i, jeg, det, at , en
    nltk.download('stopwords')

    print(stopwords.words("danish"))
    # Remove coma and full stop
    # Make a list of these words

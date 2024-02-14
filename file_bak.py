import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

from nltk.stem.snowball import SnowballStemmer

# Initialize stemmers for both German and English
german_stemmer = SnowballStemmer("german")
english_stemmer = SnowballStemmer("english")

def stem(word, language='english'):
    if language == 'german':
        return german_stemmer.stem(word.lower())
    else:
        return english_stemmer.stem(word.lower())

# Example usage
german_words = ["keine", "keiner"]
english_words = ["running", "jokes"]

german_stemmed = [stem(w, 'german') for w in german_words]
english_stemmed = [stem(w) for w in english_words]

print(german_stemmed)  # ['kein', 'kein']
print(english_stemmed)  # ['run', 'joke']

##########################################################

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    pass


a = "Tell me a joke!"
print(a)
a = tokenize(a)
print(a)

words = ["Organize", "organizes"]
stemmed_words = [stem(w) for w in words]
print(stemmed_words)





# Download NLTK data (e.g., tokenizers, corpora, etc.)
# This is a one-time setup step for downloading necessary NLTK datasets and models.
# nltk.download('popular')
# nltk.download('punkt') # pre-trained tokeniser
################################################################



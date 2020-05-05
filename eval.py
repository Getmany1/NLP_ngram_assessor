from nltk import word_tokenize, sent_tokenize 
from nltk.util import ngrams

def eval(text, n):

    #Tokenize
    text = sent_tokenize(text.lower())
    #text = [bigrams(word_tokenize(sent)) for sent in text]

    #ngrams = [list(ngrams(word_tokenize(sent), n)) for sent in sent_tokenize(text)]

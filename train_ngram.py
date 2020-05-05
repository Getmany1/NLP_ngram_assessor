from nltk import word_tokenize, sent_tokenize 
from nltk.util import ngrams, pad_sequence
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE, WittenBellInterpolated

def train_ngram(corpus, n):
    """
    Train ngram language model from a corpus.
    """

    #Read the corpus file and lowercase the text
    with open(corpus, encoding='utf8') as f:
        text = f.read().lower()

    #Tokenize
    text = sent_tokenize(text)
    text = [word_tokenize(sent) for sent in text]

    #Train ngram language model
    #Do not apply any ngram smoothing thechniques for the model
    train_data, vocab = padded_everygram_pipeline(n, text)
    lm = MLE(n)
    lm.fit(train_data, vocab)

    return lm


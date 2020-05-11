from nltk import word_tokenize, sent_tokenize 
from nltk.util import ngrams, pad_sequence
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE, WittenBellInterpolated
import dill as pickle
import os

def train_ngram(corpus, n, words=True):
    """
    Train ngram language model from a corpus.
    """

    # Read the corpus file
    with open(os.path.join('data','corpora',corpus), encoding='utf8') as f:
        text = f.read()

    # Lowercase if the model will be trained on words (to be skipped for 
    # POS tags)
    if words:
        text = text.lower()

    # Tokenize
    text = sent_tokenize(text)
    text = [word_tokenize(sent) for sent in text]

    # Train ngram language model
    # Do not apply any ngram smoothing thechniques for the model
    train_data, vocab = padded_everygram_pipeline(n, text)
    lm = MLE(n)
    lm.fit(train_data, vocab)

    # Save the model
    with open(os.path.join('data', 'models', corpus[:-4] + '.pkl'), 'wb') as f:
        pickle.dump(lm, f)

    return lm


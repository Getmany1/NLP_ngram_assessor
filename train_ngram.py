from nltk import word_tokenize, sent_tokenize
from nltk.util import ngrams, pad_sequence
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE, WittenBellInterpolated
import dill as pickle
import os

def train_ngram(corpus, n, words=True):
    """
    Train ngram (POS) language model from a corpus.
    """
    
    # Read the corpus file
    if corpus[-4:] == '.txt':
        with open(os.path.join('data','corpora',corpus), encoding='utf8') as f:
            text = f.read()

    elif corpus[-4:] == '.pkl':
        with open(os.path.join('data','corpora',corpus), 'rb') as f:
            text = pickle.load(f)

    # Lowercase if the model will be trained on words (to be skipped for 
    # POS tags)
    if words:
        if type(text) is list:
            for sent_idx, sent in enumerate(text):
                for word_idx, word in enumerate(sent):
                    text[sent_idx][word_idx] = word.lower()
        elif type(text) is str:
            text = text.lower()

    # Tokenize
    if type(text) is str:
        text = sent_tokenize(text)
        if words:
            text = [word_tokenize(sent) for sent in text]
        else:
            text = [sent.split() for sent in text]

    # Train ngram language model
    # Do not apply any ngram smoothing thechniques for the model
    train_data, vocab = padded_everygram_pipeline(n, text)
    lm = MLE(n)
    lm.fit(train_data, vocab)

    # Save the model
    with open(os.path.join('data', 'models', corpus[:-4] + '_' + str(n) + 'gram' + '.pkl'), 'wb') as f:
        pickle.dump(lm, f, 4)

    return lm
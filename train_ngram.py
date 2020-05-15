from nltk import word_tokenize, sent_tokenize
from nltk.util import ngrams, pad_sequence
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE, WittenBellInterpolated
import dill as pickle
#import pickle5
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
        #pickle5.dump(lm, f, 4) #or pickle5.HIGHEST_PROTOCOL
        pickle.dump(lm, f, 4)
    #hkl.dump(lm, os.path.join('data', 'models', corpus[:-4] + '_' + str(n) + 'gram' + '.hkl'), mode='w')

    return lm
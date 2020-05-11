from eval import tag
import itertools
from train_crf_pos import get_features
import os
from nltk import word_tokenize, sent_tokenize

def word2pos(corpus, pos_tagger):

    # Read the corpus file
    with open(os.path.join('data','corpora', corpus), encoding='utf8') as f:
        text = f.read()
        
    # Tokenize
    text = sent_tokenize(text)
    text = [word_tokenize(sent) for sent in text]

    # Convert words to tags
    features = [get_features([word for word in sent]) for sent in text]
    tags = pos_tagger.predict(features)

    # Save the corpus with POS tags instead of words
    with open(os.path.join('data','corpora', corpus[:-4] + '_pos' + '.txt'), 'w', encoding='utf-8') as f:
            f.write(' '.join(list(itertools.chain.from_iterable(tags))))
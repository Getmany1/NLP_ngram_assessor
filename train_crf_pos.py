import os
import nltk
#nltk.download('treebank')
from sklearn_crfsuite import CRF, metrics
import dill as pickle

def get_features(sent):
    return [{
        'word': sent[idx],
        'bias': True,
        'suff4': sent[idx].lower()[-4:],
        'suff3': sent[idx].lower()[-3:],
        'suff2': sent[idx].lower()[-2:],
        'suff1': sent[idx].lower()[-1:],
        'pref1': sent[idx].lower()[:1],
        'pref2': sent[idx].lower()[:2],
        'pref3': sent[idx].lower()[:3],
        'pref4': sent[idx].lower()[:4],
        'capitalized': sent[idx].istitle(),
        'all_upper': sent[idx].isupper(), # is_acronym
        'first': idx == 0,
        'last': idx == len(sent)-1,
        'prev_word': sent[idx-1] if idx>0 else '',
        'prev_prev_word': sent[idx-2] if idx>1 else '',
        'next_word': sent[idx+1] if idx<len(sent)-1 else '',
        'next_next_word': sent[idx+2] if idx<len(sent)-2 else '',
        'has_num': any(l.isdigit() for l in sent[idx]),
        'has_hyphen': '-' in sent[idx],
    } for idx in range(len(sent))]

def train_crf_pos(corpus):
    #language = 'english'

    # Required corpus structure:
    # [[(w1,t1), (w2,t2),...(wn,tn)], [(w1,t1)(w2,t2),...(wm,tm)],...]
    #corpus_name = 'Penn_treebank'
    corpus_name = 'UD_Swedish-Talbanken'
    #corpus = nltk.corpus.treebank.tagged_sents()

    #feat_all = {} # common features (baseline set)
    #feat_en = {} # extra features for English
    #features = {**feat_all, **feat_en}
    train_frac = 0.9 # fraction of data for the training set
    split_idx = int(train_frac*len(corpus))

    # Extract the feautures and separate labels from features
    X = [get_features([pair[0] for pair in sent]) for sent in corpus]
    y = [[pair[1] for pair in sent] for sent in corpus]

    # Create the training and the test sets
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    # Create the CRF model
    model = CRF(
        algorithm='lbfgs', # gradient descent using the L-BFGS method
        c1=0.1, # coeff. for L1 regularization
        c2=0.1, # coeff. for L2 regularization
        max_iterations=100,
    )

    # Train the model
    model.fit(X_train, y_train)

    # Save the model
    with open(os.path.join('data', 'models', corpus_name + '_crf.pkl'), 'wb') as f:
        pickle.dump(model, f)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Test accuracy: %.4f" % metrics.flat_accuracy_score(y_test, y_pred))

    return model



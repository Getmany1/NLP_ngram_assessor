import nltk
#nltk.download('treebank')
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

language = 'english'
corpus = nltk.corpus.treebank.tagged_sents()
# common features (baseline set)
feat_all = {} 
feat_en = {} # extra features for English
features = {**feat_all, **feat_en}
test_frac = 0.8 # fraction of data for the training set

X = [get_features([pair[0] for pair in sent]) for sent in corpus]
y = [[pair[1] for pair in sent] for sent in corpus]

X_train = X[:int(test_frac*len(corpus))]
y_train = y[:int(test_frac*len(corpus))]
X_test = X[int(test_frac*len(corpus)):]
y_test = y[int(test_frac*len(corpus)):]

print(len(corpus))
print(len(X))
print(X[0])


#trainset = corpus[:int(test_frac*len(corpus))]
#testset = corpus[int(test_frac*len(corpus)):]


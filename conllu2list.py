import os
from conllu import parse
import pickle

def conllu2list():
    """
    Convert UD_Swedish-Talbanken conllu files to a single corpus for training 
    the POS tagger.
    """
    data_file_1 = os.path.join('data','corpora','UD_Swedish-Talbanken','sv_talbanken-ud-train.conllu')
    data_file_2 = os.path.join('data','corpora','UD_Swedish-Talbanken','sv_talbanken-ud-test.conllu')
    data_file_3 = os.path.join('data','corpora','UD_Swedish-Talbanken','sv_talbanken-ud-dev.conllu')
    sentences = []
    corpus = []
    
    # Read conllu files
    with open(data_file_1, 'r', encoding='utf8') as f:
        data = f.read()
    sentences.extend(parse(data))
    with open(data_file_2, 'r', encoding='utf8') as f:
        data = f.read()
    sentences.extend(parse(data))
    with open(data_file_3, 'r', encoding='utf8') as f:
        data = f.read()
    sentences.extend(parse(data))
    
    # Extract tokens and POS tags
    for sentence in sentences:
        sent = []
        for token in sentence:
            sent.append((token['form'], token['upostag']))
        corpus.append(sent)
    
    # Save the corpus
    with open(os.path.join('data','corpora','UD_Swedish-Talbanken.pkl'), 'wb') as f:
        pickle.dump(corpus, f, 4)
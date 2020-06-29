#from eval import tag
import itertools
from train_crf_pos import get_crf_features
import os
from nltk import word_tokenize, sent_tokenize
import dill as pickle
from conllu import parse

def morph_segment(corpus, morph_model):
    """
    Divide the words in the corpus into segments and save as a separate corpus
    """
    text_segmented = []
    
    # Read the corpus file
    if corpus[-4:] == '.txt':
        with open(os.path.join('data','corpora',corpus), encoding='utf8') as f:
            text = f.read()

    elif corpus[-4:] == '.pkl':
        with open(os.path.join('data','corpora',corpus), 'rb') as f:
            text = pickle.load(f)
            
    if type(text) is str:
        text = text.lower()
        text = sent_tokenize(text)
        text = [word_tokenize(sent) for sent in text]

    for sent_idx, sent in enumerate(text):
        sent_segmented = []
        for word_idx, word in enumerate(sent):
            [sent_segmented.append(segment) for segment in
             morph_model.viterbi_nbest(word.lower(),1)[0][0]]
        text_segmented.append(sent_segmented)
    
    # Save the corpus
    with open(os.path.join('data','corpora', corpus[-4:] + '_morph_segmented.pkl'), 'wb') as f:
        pickle.dump(corpus, f, 4)
        
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

def vrt2lists():
    """
    Convert Swedish Yle .vrt corpus files to a tokenized text corpus and 
    tokenized corresponding POS tag corpus.
    corpus = [[w1,w2,...], [w1,w2,...], ...]
    tag_corpus = [[t1,t2,...], [t1,t2,...], ...]
    """
    corpus_folder = os.path.join('data', 'corpora', 'ylenews-sv-2012-2018-s-vrt',
                                 'vrt')
    corpus = []
    tag_corpus = []
    files = list(os.walk(corpus_folder))[0][2]
    for file in files:
        with open(os.path.join(corpus_folder, file), encoding='utf8') as f:
            data = f.read().split('</sentence>')
        for sent in data:
            sentence = []
            tag_sentence = []
            items = [element.split('\t') for element in sent.split('\n')]
            for item in items:
                if len(item) == 8:
                    word = item[0]
                    tag = item[3]
                    #sentence.append((word, tag))
                    sentence.append(word)
                    tag_sentence.append(tag)
            if len(sentence) > 1 and len(sentence) == len(tag_sentence):
                corpus.append(sentence)
                tag_corpus.append(tag_sentence)

    
    # Save the corpora
    with open(os.path.join('data','corpora','Yle_sv.pkl'), 'wb') as f:
        pickle.dump(corpus, f, 4)
        
    with open(os.path.join('data','corpora','Yle_sv_pos.pkl'), 'wb') as f:
        pickle.dump(tag_corpus, f, 4)

    #with open(os.path.join('data','corpora','Yle_sv_words_tags.pkl'), 'wb') as f:
        #pickle.dump(corpus, f, 4)
        
def vrt2lists_fi():
    """
    Convert Finnish Wikipedia .vrt corpus files to a tokenized text corpus and 
    tokenized corresponding POS tag corpus.
    corpus = [[w1,w2,...], [w1,w2,...], ...]
    tag_corpus = [[t1,t2,...], [t1,t2,...], ...]
    """
    corpus_folder = os.path.join('data', 'corpora', 'wikipedia-fi-2017-src',
                                 'wikipedia-fi-2017-src')
    corpus = []
    tag_corpus = []
    files = list(os.walk(corpus_folder))[0][2]
    for file in files:
        with open(os.path.join(corpus_folder, file), encoding='utf8') as f:
            data = f.read().split('</sentence>')
        for sent in data:
            sentence = []
            tag_sentence = []
            items = [element.split('\t') for element in sent.split('\n')]
            for item in items:
                if len(item) == 10:
                    word = item[1]
                    tag = item[3]
                    #sentence.append((word, tag))
                    sentence.append(word)
                    tag_sentence.append(tag)
            if len(sentence) > 1 and len(sentence) == len(tag_sentence):
                corpus.append(sentence)
                tag_corpus.append(tag_sentence)

    
    # Save the corpora
    with open(os.path.join('data','corpora','Wikipedia_fi_2017.pkl'), 'wb') as f:
        pickle.dump(corpus, f, 4)
        
    with open(os.path.join('data','corpora','Wikipedia_fi_2017_pos.pkl'), 'wb') as f:
        pickle.dump(tag_corpus, f, 4)

    #with open(os.path.join('data','corpora','Wikipedia_fi_2017_words_tags.pkl'), 'wb') as f:
        #pickle.dump(corpus, f, 4)

def word2pos(corpus, pos_tagger):

    # Read the corpus file
    with open(os.path.join('data','corpora', corpus), encoding='utf8') as f:
        text = f.read()
    # Load the POS tagger
    with open(os.path.join('data', 'models', pos_tagger), 'rb') as f:
        pos_tagger = pickle.load(f)
        
    # Tokenize
    text = sent_tokenize(text)
    text = [word_tokenize(sent) for sent in text]

    # Convert words to tags
    features = [get_crf_features([word for word in sent]) for sent in text]
    tags = pos_tagger.predict(features)

    # Save the corpus with POS tags instead of words
    with open(os.path.join('data','corpora', corpus[:-4] + '_pos_suc' + '.txt'), 'w', encoding='utf-8') as f:
            f.write(' '.join(list(itertools.chain.from_iterable(tags))))
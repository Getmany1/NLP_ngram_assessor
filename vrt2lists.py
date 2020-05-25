import os
import dill as pickle

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

    with open(os.path.join('data','corpora','Yle_sv_words_tags.pkl'), 'wb') as f:
        pickle.dump(corpus, f, 4)
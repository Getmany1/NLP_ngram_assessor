import os
from gensim.corpora.wikicorpus import WikiCorpus

def make_sv_wikicorpus():
    wiki = WikiCorpus(os.path.join('data','corpora','swwiki-latest-pages-articles.xml.bz2'), 
                      lemmatize=False, dictionary={})
    texts_num = 0
    with open(os.path.join('data','corpora', 'wikipedia_sv.txt'), 'w', encoding='utf-8') as output:
         for text in wiki.get_texts():
             output.write(' '.join(text) + '\n')
             texts_num += 1
             if texts_num % 10000 == 0:
                 logging.info("Parsed %d th articles" % texts_num)
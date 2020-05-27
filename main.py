import os
import nltk
import dill as pickle
from train_ngram import train_ngram
from train_crf_pos import train_crf_pos
from eval import eval
from pos_descriptions import *

# Parameters
lang = 'swedish'
corp_dir = os.path.join('data', 'corpora')
model_dir = os.path.join('data', 'models')

# Text corpora for training LMs
# ENGLISH
#lm_corpus_name = 'wikipedia2008_en.txt'
# SWEDISH
lm_corpus_name = 'wikipedia_sv.txt' #source: https://linguatools.org/
#lm_corpus_name = 'Yle_sv.pkl' #source: Kielipankki

# POS-tagged corpora for training POS taggers
# ENGLISH
#pos_corpus = nltk.corpus.treebank.tagged_sents()
#pos_corpus_name = 'Penn_treebank'
# SWEDISH
pos_corpus = 'UD_Swedish-Talbanken.pkl' #source: https://universaldependencies.org/
#pos_corpus = 'Yle_sv_words_tags.pkl

# Language Models
# ENGLISH
#lm_name = 'iltalehti_2gram.pkl'
#lm_name = 'wikipedia2008_en_3gram.pkl'
#lm_name = 'wikipedia2008_en_2gram.pkl'
# SWEDISH
#lm_name = 'wikipedia_sv_2gram.pkl'
#lm_name = 'wikipedia_sv_3gram.pkl'
lm_name = 'Yle_sv_2gram.pkl'

# POS Language Models
# ENGLISH
#pos_lm_name = 'wikipedia2008_en_pos_3gram.pkl'
# SWEDISH
pos_lm_name = 'wikipedia_sv_pos_3gram.pkl'
#pos_lm_name = 'Yle_sv_pos_3gram.pkl'

# POS Taggers
# ENGLISH
#pos_name = 'Penn_treebank_crf.pkl'
# SWEDISH
pos_name = 'UD_Swedish-Talbanken_crf.pkl'
#pos_name = 'Yle_sv_pos_crf.pkl'

lm_type = 'ngram' # language model type
n = 2 # ngram size
pos_type = 'crf' # POS model type
threshold = float('-inf') # lowest threshold for ngram log-probability
                            # in text evaluation
#text_to_analyze = "This is a test text. The automatic assessor will report OOV words and uncommon ngrams."
text_to_analyze = 'Projektet DigiTala har som målsättning att analysera, utveckla och pröva möjligheter att testa muntlig färdighet med elektriska och datorbaserade medel. Oavsett regleringen i gymnasieskolans styrdokument att beakta samtliga kommunikativa delfärdigheter, saknas det muntliga testet fortfarande i den finländska studentexamen.'
result_file = 'testresult'
TRAIN_LM = False # train new language model or load pretrained one
TRAIN_POS = False # train POS tagger or load pretrained one
SAVE_REPORT = False # save evaluation results

if TRAIN_LM:
    if lm_type == 'ngram':
        lm = train_ngram(lm_corpus_name, n, words=True)
else:
    with open(os.path.join(model_dir, lm_name), 'rb') as f:
        lm = pickle.load(f)

if TRAIN_POS:
    if pos_type == 'crf':
        if pos_corpus[-4:] == '.pkl':
            pos_corpus_name = pos_corpus[:-4]
            with open(os.path.join(corp_dir, pos_corpus), 'rb') as f:
                pos_corpus = pickle.load(f)
        pos_tagger = train_crf_pos(pos_corpus, pos_corpus_name)
else:
    with open(os.path.join(model_dir, pos_name), 'rb') as f:
        pos_tagger = pickle.load(f)

with open(os.path.join(model_dir, pos_lm_name), 'rb') as f:
    pos_lm = pickle.load(f)

# Load the descriptions of POS tags
if pos_name == 'Penn_treebank_crf.pkl':
    pos_descr_dict = pos_dict_en()
elif pos_name == 'UD_Swedish-Talbanken_crf.pkl':
    pos_descr_dict = pos_dict_sv()
elif pos_name == 'Yle_sv_pos_crf.pkl':
    pos_descr_dict = pos_dict_sv_suc()

eval_result = eval(text_to_analyze, lm, pos_lm, pos_tagger, pos_descr_dict, threshold)
if SAVE_REPORT:
    with open(os.path.join('results', result_file), 'w', encoding='utf-8') as f:
            f.write(eval_result)
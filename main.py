import os
import dill as pickle
from train_ngram import train_ngram
from train_crf_pos import train_crf_pos
from train_morfessor import train_morfessor
from eval import eval
from pos_descriptions import *
import morfessor
from stanza import Pipeline

# Parameters
lang = 'swedish'
corp_dir = os.path.join('data', 'corpora')
model_dir = os.path.join('data', 'models')

# Text corpora for training LMs
# ENGLISH
#lm_corpus_name = 'wikipedia2008_en.txt'
# SWEDISH
#lm_corpus_name = 'wikipedia_sv.txt' #source: https://linguatools.org/
#lm_corpus_name = 'Yle_sv.pkl' #source: Kielipankki
# FINNISH
lm_corpus_name = 'Wikipedia_fi_2017.pkl' #source: Kielipankki

# Text corpora for training POS LMs
# ENGLISH
#pos_lm_corpus_name = 'wikipedia2008_en_pos.txt'
# SWEDISH
#pos_lm_corpus_name = 'Yle_sv_pos.pkl'
#pos_lm_corpus_name = 'wikipedia_sv_pos.txt'
# FINNISH
pos_lm_corpus_name = 'Wikipedia_fi_2017_pos.pkl'

# Text corpora for training Morfessor models
# SWEDISH
#morph_corpus = 'yle_sv_minicorpus.txt'
#morph_corpus = 'Yle_sv.txt'
# FINNISH
morph_corpus = 'Wikipedia_fi_2017.txt'

# POS-tagged corpora for training POS taggers
# ENGLISH
#pos_corpus = nltk.corpus.treebank.tagged_sents()
#pos_corpus_name = 'Penn_treebank'
# SWEDISH
#pos_corpus = 'UD_Swedish-Talbanken.pkl' #source: https://universaldependencies.org/
#pos_corpus = 'Yle_sv_words_tags.pkl
# FINNISH
pos_corpus = 'Wikipedia_fi_2017_words_tags.pkl'

# Language Models
# ENGLISH
#lm_name = 'wikipedia2008_en_3gram.pkl'
#lm_name = 'wikipedia2008_en_2gram.pkl'
# SWEDISH
#lm_name = 'wikipedia_sv_2gram.pkl'
#lm_name = 'wikipedia_sv_3gram.pkl'
#lm_name = 'Yle_sv_2gram.pkl'
# FINNISH
#lm_name = 'iltalehti_2gram.pkl'
lm_name = 'Wikipedia_fi_2017_2gram.pkl'
#lm_name = 'Wikipedia_fi_2017_3gram.pkl'

# POS Language Models
# ENGLISH
#pos_lm_name = 'wikipedia2008_en_pos_3gram.pkl'
# SWEDISH
#pos_lm_name = 'wikipedia_sv_pos_3gram.pkl'
#pos_lm_name = 'Yle_sv_pos_3gram.pkl'
# FINNISH
pos_lm_name = 'Wikipedia_fi_2017_pos_3gram.pkl'

# Morfessor models
# SWEDISH
#morph_model = 'yle_sv_minicorpus_morph'
#morph_model = 'Yle_sv_morph'
# FINNISH
morph_model = 'Wikipedia_fi_2017_morph'

# POS Taggers
# ENGLISH
#pos_name = 'Penn_treebank_crf.pkl'
# SWEDISH
#pos_name = 'UD_Swedish-Talbanken_crf.pkl'
#pos_name = 'Yle_sv_pos_crf.pkl'
# FINNISH
pos_name = 'Wikipedia_fi_2017_words_tags_crf.pkl'

# POS processors for extracting morphological features
# SWEDISH
#nlp = Pipeline(lang='sv', processors='tokenize,mwt,pos') #source: https://stanfordnlp.github.io/stanza/
# FINNISH
nlp = Pipeline(lang='fi', processors='tokenize,mwt,pos') #source: https://stanfordnlp.github.io/stanza/

lm_type = 'ngram' # language model type
pos_lm_type = 'ngram' # POS language model type
n = 2 # ngram size for LM
n_pos = 3 # ngram size for POS LM
split_prob = 0.5 # split probability for train_morfessor()
pos_type = 'crf' # POS model type
threshold = float('-inf') # lowest threshold for ngram log-probability
                            # in text evaluation
#text_to_analyze = "This is a test text. The automatic assessor will report OOV words and uncommon ngrams."
#text_to_analyze = 'Projektet DigiTala har som målsättning att analysera, utveckla och pröva möjligheter att testa muntlig färdighet med elektriska och datorbaserade medel. Oavsett regleringen i gymnasieskolans styrdokument att beakta samtliga kommunikativa delfärdigheter, saknas det muntliga testet fortfarande i den finländska studentexamen.'
text_to_analyze = 'DigiTala on poikkitieteellinen tutkimushanke, jonka tavoitteena on kehittää tietokoneavusteinen suullisen kielitaidon koe lukion päättövaiheeseen. '
result_file = 'testresult'

TRAIN_LM = False # train new language model or load pretrained one
TRAIN_POS_LM = False # train new POS language model or load pretrained one
TRAIN_POS = False # train POS tagger or load pretrained one
TRAIN_MORPH = False # Train Morfessor model or load pretrained one
SAVE_REPORT = False # save evaluation results

if TRAIN_LM:
    if lm_type == 'ngram':
        lm = train_ngram(lm_corpus_name, n, words=True)
else:
    with open(os.path.join(model_dir, lm_name), 'rb') as f:
        lm = pickle.load(f)

if TRAIN_POS_LM:
    if pos_lm_type == 'ngram':
        pos_lm = train_ngram(pos_lm_corpus_name, n_pos, words=False)
else:
    with open(os.path.join(model_dir, pos_lm_name), 'rb') as f:
        pos_lm = pickle.load(f)  
    
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
    
if TRAIN_MORPH:
    morph_model = train_morfessor(morph_corpus, split_prob)
else:
    io = morfessor.MorfessorIO(compound_separator=r"[^-\w]+" ,lowercase=True)
    morph_model = io.read_binary_model_file(os.path.join(model_dir, morph_model))

# Load the descriptions of POS tags
if pos_name == 'Penn_treebank_crf.pkl':
    pos_descr_dict = pos_dict_en()
elif pos_name == 'UD_Swedish-Talbanken_crf.pkl' or pos_name == 'Wikipedia_fi_2017_words_tags_crf.pkl':
    pos_descr_dict = pos_dict_sv()
elif pos_name == 'Yle_sv_pos_crf.pkl':
    pos_descr_dict = pos_dict_sv_suc()

eval_result = eval(text_to_analyze, lm, pos_lm, pos_tagger, morph_model, nlp, 
                   pos_descr_dict, threshold)
if SAVE_REPORT:
    with open(os.path.join('results', result_file), 'w', encoding='utf-8') as f:
            f.write(eval_result)
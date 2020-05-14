import os
import nltk
import dill as pickle
from train_ngram import train_ngram
from train_crf_pos import train_crf_pos
from eval import eval

# Parameters
corp_dir = os.path.join('data', 'corpora')
model_dir = os.path.join('data', 'models')
lm_corpus_name = 'wikipedia2008_en.txt'
pos_corpus = nltk.corpus.treebank.tagged_sents()

#lm_name = 'iltalehti.pkl'
lm_name = 'wikipedia2008_en.pkl'
pos_lm_name = 'wikipedia2008_en_pos.pkl'
pos_name = 'Penn_treebank_crf.pkl'

lm_type = 'ngram' # language model type
n = 2 # ngram size
pos_type = 'crf' # POS model type
threshold = float('-inf') # lowest threshold for ngram log-probability
                            # in text evaluation
text_to_analyze = "This is a test text. The automatic assessor will report OOV words and uncommon ngrams."
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
        pos_tagger = train_crf_pos(pos_corpus)
else:
    with open(os.path.join(model_dir, pos_name), 'rb') as f:
        pos_tagger = pickle.load(f)

with open(os.path.join(model_dir, pos_lm_name), 'rb') as f:
    pos_lm = pickle.load(f)

eval_result = eval(text_to_analyze, lm, pos_lm, pos_tagger, threshold)
if SAVE_REPORT:
    with open(os.path.join('results', result_file), 'w', encoding='utf-8') as f:
            f.write(eval_result)
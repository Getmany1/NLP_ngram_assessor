import os
import dill as pickle
from train_ngram import train_ngram
from eval import eval

#Parameters
corp_dir = os.path.join('data', 'corpora')
model_dir = os.path.join('data', 'models')
corpus_name = 'wikipedia2008_en.txt'
model_name = 'iltalehti.pkl'
model_type = 'ngram' #model type
n = 2 #ngram size
threshold = float('-inf') #lowest threshold for ngram log-probability
                            #in text evaluation
text_to_analyze = "This is a test text. The automatic assessor will report \
    OOV words and uncommon ngrams."
result_file = "testresult"
TRAIN = False #train new model or load pretrained one
SAVE_REPORT = False #save evaluation results

if TRAIN:
    if model_type == 'ngram':
        lm = train_ngram(corpus_name, n)
else:
    with open(os.path.join(model_dir, model_name), 'rb') as f:
        lm = pickle.load(f)

eval_result = eval(text_to_analyze, n, lm, threshold)
if SAVE_REPORT:
    with open(os.path.join("results", result_file), 'w', encoding='utf-8') as f:
            f.write(eval_result)
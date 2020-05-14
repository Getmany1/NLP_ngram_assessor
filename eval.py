from nltk import word_tokenize, sent_tokenize
from train_crf_pos import get_features
from nltk.util import ngrams
from nltk.lm.preprocessing import pad_both_ends
import copy
import itertools
import numpy as np
from collections import Counter 
from pos_descriptions import *

def tag(text, pos_tagger):
    """
    Tag text with parts of speech
    """
    features = [get_features([word for word in sent]) for sent in text]
    tags = pos_tagger.predict(features)
    tagged_text = []
    for i in range(len(text)):
        tagged_sent = []
        for j in range(len(text[i])):
            tagged_sent.append((text[i][j], tags[i][j]))
        tagged_text.append(tagged_sent)
    #print(tags)
    return tags, tagged_text
    
def most_probable_tag(context, k, pos_lm):
    """
    Compute k most likely next tags
    """
    #pos_lm.counts.__getitem__(['DT', 'JJ']).max() #next item after the context
    prob_dict = {}
    for next_tag in pos_lm.vocab:
        prob_dict[next_tag] = pos_lm.logscore(next_tag,context)
    #return #max(prob_dict, key=prob_dict.get)
    return Counter(prob_dict).most_common(k)

def eval(text_to_analyze, lm, pos_lm, pos_tagger, threshold=float('-inf')):
    
    #
    lang = 'english'
    if lang == 'english':
        pos_descr_dict = pos_dict_en()
    
    n = lm.counts.__len__() # LM ngram order
    n_pos = pos_lm.counts.__len__() # LM ngram order
    
    # Tokenize
    text = sent_tokenize(text_to_analyze)
    text = [word_tokenize(sent) for sent in text]

    # Add POS tags
    tags, tagged_text = tag(text, pos_tagger)
    tags = [list(pad_both_ends(sent,n_pos)) for sent in tags]

    # Lowercase
    text = [[word.lower() for word in sent] for sent in text]

    # Highlight OOV words with *UNK* ('word' -> 'word*<UNK>')
    errs_unk = ""
    err_unk_count = 0
    unk_mark = '<UNK>'
    for sent_idx, sent in enumerate(text):
        for word_idx, word in enumerate(sent):
            if word not in lm.vocab:
                text[sent_idx][word_idx] = word+'*'+unk_mark
                err_unk_count += 1
                tag_to_use = most_probable_tag(tags[sent_idx]
                                               [word_idx-2+(n_pos-1):word_idx+(n_pos-1)], 
                                               1, pos_lm)[0][0]
                errs_unk += str(err_unk_count) + '. ' + \
                    word + ': try to replace with some ' + \
                    pos_descr_dict[tag_to_use].lower() +\
                            '.\n'
    
    # Add start-of-sentence and end-of-sentence symbols (<s> and </s>)
    text = [list(pad_both_ends(sent,n)) for sent in text]
    #tags = [list(pad_both_ends(sent,n_pos)) for sent in tags]

    #lm.vocab.lookup(text, unk_cutoff=1)
    #lm.counts.__getitem__(['automatic']).max()

    evaluated = copy.deepcopy(text)
    errs = ""
    err_count = 0

    # Analyze ngrams in the given text. Mark ngrams with low
    # log-probability with '*error_number' ('w1, ..., wn' -> 
    # 'w1, ..., wn*err_number') and write down such ngrams after the text
    # with corresponding error numbers. Ngrams with OOV words can be skipped,
    # because they have 0 probability in unsmoothed language model (LM) or
    # very low probability (supposed to be far below the threshold)
    # in smoothed LM.
    for sent_idx, sent in enumerate(text):
        i=0
        while i < len(sent)-n+1:
            ngram = sent[i:i+n]
            if all([unk_mark not in word for word in ngram]):
                if lm.logscore(ngram[-1], ' '.join(ngram[:-1]).split()) <= threshold:
                    err_count += 1
                    next_word = ', '.join([pair[0] for pair in 
                                 Counter(lm.counts.__getitem__(ngram[:-1])).most_common(3)])
                    evaluated[sent_idx][i+n-1] += '*' + str(err_count)
                    errs += str(err_count) + '. ' + \
                        ' '.join(text[sent_idx][i:i+n]) + ': try to replace the word '\
                            +ngram[-1] + ' with some other word like ' + next_word + '.\n'
            i += 1

    
    # Print the evaluation results
    print("Text to evaluate: ")
    print(text_to_analyze)
    print()
    print("(For evaluator) Text tagged with part-of-speech tags: ")
    print(tagged_text)
    print()
    print("Analyzed text with annotations: ")
    #print(' '.join(list(itertools.chain(*evaluated))))
    print(' '.join(list(itertools.chain(*[sent[n-1:-(n-1)] for sent in evaluated]))))
    print()
    print("Unknown words: ")
    print(errs_unk)
    print("Uncommon ngrams: ")
    print(errs)
    
    # Return the results
    result = "Text to evaluate:\n " + text_to_analyze + "\n " + \
        "(For evaluator) Text tagged with part-of-speech tags:\n" + \
            str(tagged_text) + "\n" + "Analyzed text with annotations:\n" + \
                ' '.join(list(itertools.chain(*[sent[n-1:-(n-1)] for sent in evaluated]))) + "\n" + \
                    "Unknown words:\n" + errs_unk + "Uncommon ngrams:\n" + errs
    return result
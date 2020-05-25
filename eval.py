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
    
'''def most_probable_tag(context, k, pos_lm):
    """
    Compute k most likely next tags
    """
    #pos_lm.counts.__getitem__(['DT', 'JJ']).max() #next item after the context
    prob_dict = {}
    for next_tag in pos_lm.vocab:
        prob_dict[next_tag] = pos_lm.logscore(next_tag,context)
    #return #max(prob_dict, key=prob_dict.get)
    return Counter(prob_dict).most_common(k)'''

def mean_logprob(context, pos_lm):
    logprobs = []
    for tag in pos_lm.vocab:
        logprob = pos_lm.logscore(tag,context)
        if logprob != -float('inf'):
            logprobs.append(logprob)
    return np.mean(logprobs)    

def eval(text_to_analyze, lm, pos_lm, pos_tagger, threshold=float('-inf')):
    
    #
    lang = 'swedish'
    # Load the descriptions of POS tags
    if lang == 'english':
        pos_descr_dict = pos_dict_en()
    elif lang == 'swedish':
        pos_descr_dict = pos_dict_sv_suc()
    
    n = 2 # LM ngram order. Use bigrams
    n_pos = pos_lm.counts.__len__() # POS LM ngram order
    
    # Tokenize
    text = sent_tokenize(text_to_analyze)
    text = [word_tokenize(sent) for sent in text]

    # Add POS tags
    tags, tagged_text = tag(text, pos_tagger)
    
    # Pad tag sequences with 
    # start-of-sentence and end-of-sentence symbols (<s> and </s>)
    tags = [list(pad_both_ends(sent,n_pos)) for sent in tags]

    # Lowercase the text
    text = [[word.lower() for word in sent] for sent in text]

    # Highlight OOV words with *UNK* ('word' -> 'word*<UNK>')
    # Predict the most likely POS tag t_i for the unkwown word w_i based on the
    # sequence of previous tags t_(i-2), t_(i-1) using the POS LM
    errs_unk = ""
    err_unk_count = 0
    unk_mark = '<UNK>'
    for sent_idx, sent in enumerate(text):
        for word_idx, word in enumerate(sent):
            if word not in lm.vocab:
                text[sent_idx][word_idx] = word+'*'+unk_mark
                err_unk_count += 1
                errs_unk += str(err_unk_count) + '. ' + word
                next_tag_dict = pos_lm.counts.__getitem__(tags[sent_idx]
                                                           [word_idx-2+(n_pos-1):
                                                            word_idx+(n_pos-1)])
                if len(next_tag_dict)>0:
                    tag_to_use = next_tag_dict.max()
                    errs_unk += ': try to replace with some ' + \
                        pos_descr_dict[tag_to_use].lower() +\
                                '.\n'
                else:
                    errs_unk += '??? \n'
    
    # Pad the sentences with start-of-sentence and end-of-sentence symbols
    text = [list(pad_both_ends(sent,n)) for sent in text]

    #lm.vocab.lookup(text, unk_cutoff=1)
    #lm.counts.__getitem__(['automatic']).max()

    # Analyze ngrams in the given text. Mark ngrams with low
    # log-probability with '*error_number' ('w_1, ..., w_n' -> 
    # 'w_1, ..., w_n*err_number') and write down such ngrams after the text
    # with corresponding error numbers. Ngrams with OOV words can be skipped,
    # because they have 0 probability in unsmoothed language model (LM) or
    # very low probability (supposed to be far below the threshold)
    # in smoothed LM.
    evaluated = copy.deepcopy(text)
    errs = ""
    err_count = 0
    for sent_idx, sent in enumerate(text):
        i=0
        while i < len(sent)-n+1:
            ngram = sent[i:i+n]
            if all([unk_mark not in word for word in ngram]):
                if lm.logscore(ngram[-1], ' '.join(ngram[:-1]).split()) <= threshold:
                    err_count += 1
                    
                    evaluated[sent_idx][i+n-1] += '*' + str(err_count)
                    
                    # Suggest 3 most likely words to replace the last word of 
                    # the unknown ngram based on its previous word(s)
                    #next_word = ', '.join([pair[0] for pair in 
                    #             Counter(lm.counts.__getitem__(ngram[:-1])).most_common(3)])
                    #errs += str(err_count) + '. ' + \
                    #     ' '.join(text[sent_idx][i:i+n]) + ': try to replace the word '\
                    #         +ngram[-1] + ' with some other word like ' + next_word + '.\n'
                      
                    errs += str(err_count) + '. ' + ' '.join(ngram) + ': '
                    
                    # If tag t_i cannot follow the tag sequence t_(i-2), t_(i-1)
                    # (P(t_i)|t_(i-2),t_(i-1)=0), suggest the most likely tag
                    # to replace with (argmax(P(t_i)|t_(i-2),t_(i-1))).
                    # For unknown word ngram but possible corresponding 
                    # POS ngram, compare the log probability for the tag t_i to
                    # follow the tag sequence t_(i-2), t_(i-1) to the mean of
                    # the log probabilities of all possible tags t_i that can
                    # follow the tag sequence t_(i-2), t_(i-1). If it is above
                    # the average, suppose that the tag sequence is common enough
                    # and suggest to replace the last word of the corresponding
                    # word sequence with another word belonging to the same
                    # part of speech. If it is below the average, ask to use
                    # another part of speech and suggest the most likely tag.
                    pos_logscore = pos_lm.logscore(tags[sent_idx][i +(n_pos-1)],
                                                   tags[sent_idx][i-2+(n_pos-1):
                                                         i +(n_pos-1)])
                    avg_pos_logscore = mean_logprob(tags[sent_idx][i-2+(n_pos-1):
                                                                       i +(n_pos-1)],
                                                        pos_lm)
                    if pos_logscore == -float('inf') or pos_logscore<avg_pos_logscore:
                        errs += 'you used the ' + pos_descr_dict[tags[sent_idx]
                                                                 [i +(n_pos-1)]].lower() + ' ' + \
                            ngram[-1]
                        next_tag_dict = pos_lm.counts.__getitem__(tags[sent_idx]
                                                           [i-2+(n_pos-1):
                                                            i+(n_pos-1)])
                        if len(next_tag_dict) > 0:
                            errs += '; try to use another part of speech, for example ' + \
                                pos_descr_dict[next_tag_dict.max()].lower() + \
                                    '.\n'
                        else:
                            # No tag t_i can follow the tag sequence
                            # t_(i-2), t_(i-1)
                            errs += '; ???\n'
                    else:                           
                        errs += 'try to use some other ' + pos_descr_dict[tags[sent_idx]
                                                                          [i +(n_pos-1)]].lower() + \
                            ' instead of ' + ngram[-1] + '.\n'        
            i += 1

    # Print the evaluation results
    print("Text to evaluate: ")
    print(text_to_analyze)
    print()
    print("(For human evaluator) Text tagged with part-of-speech tags: ")
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
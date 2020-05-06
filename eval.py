from nltk import word_tokenize, sent_tokenize 
from nltk.util import ngrams
from nltk.lm.preprocessing import pad_both_ends
import copy
import itertools

def eval(text_to_analyze, n, lm, threshold=float('-inf')):

    #Tokenize
    text = sent_tokenize(text_to_analyze.lower())
    text = [word_tokenize(sent) for sent in text]

    #Highlight OOV words with *UNK* ('word' -> '<UNK*word*UNK>')
    unk_mark = '<UNK>'
    for sent_idx, sent in enumerate(text):
        for word_idx, word in enumerate(sent):
            if word not in lm.vocab:
                text[sent_idx][word_idx] = word+'*'+unk_mark
    
    #Add start-of-sentence and end-of-sentence symbols (<s> and </s>)
    text = [list(pad_both_ends(sent,n)) for sent in text]

    evaluated = copy.deepcopy(text)
    errs = ""
    err_count = 0

    #Analyze ngrams in the given text. Mark ngrams with low
    #log-probability with '*error_number' ('w1, ..., wn' -> 
    #'w1, ..., wn*err_number') and write down such ngrams after the text
    #with corresponding error numbers. Ngrams with OOV words can be skipped,
    #because they have 0 probability in unsmoothed language model (LM) or
    #very low probability (supposed to be far below the threshold)
    #in smoothed LM.
    for sent_idx, sent in enumerate(text):
        i=0
        while i < len(sent)-n+1:
            ngram = sent[i:i+n]
            if all([unk_mark not in word for word in ngram]):
                if lm.logscore(ngram[-1], ' '.join(ngram[:-1]).split()) <= threshold:
                    err_count += 1
                    evaluated[sent_idx][i+n-1] += '*' + str(err_count)
                    errs += str(err_count) + '. ' + \
                        ' '.join(text[sent_idx][i:i+n]) + '\n'
            i += 1
            
    #Print the evaluation results
    print("Text to evaluate:")
    print(text_to_analyze)
    print()
    print("Analyzed text with annotations:")
    print(' '.join(list(itertools.chain(*evaluated))))
    print()
    print("Uncommon ngrams:")
    print(errs)
    
    #Return the results
    result = "Text to evaluate:\n " + text_to_analyze + "\n " + \
        "Analyzed text with annotations:\n" + \
            ' '.join(list(itertools.chain(*evaluated))) + "\n" + \
                "Uncommon ngrams:\n" + errs
    return result
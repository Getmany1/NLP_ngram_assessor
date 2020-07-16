from nltk import word_tokenize, sent_tokenize
from train_crf_pos import get_crf_features
from nltk.util import ngrams
import morfessor
from nltk.lm.preprocessing import pad_both_ends
import copy
import itertools
import numpy as np
from collections import Counter

def tag(text, pos_tagger):
    """
    Tag text with parts of speech
    """
    features = [get_crf_features([word for word in sent]) for sent in text]
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
    """
    Mean log-probability among all possible ngrams with 
    fixed n-1 POS tags
    """
    logprobs = []
    for tag in pos_lm.vocab:
        logprob = pos_lm.logscore(tag,context)
        if logprob != -float('inf'):
            logprobs.append(logprob)
    return np.mean(logprobs)

def typos(oov_word, morph_model, lm_segmented, threshold=-50):
    """
    Check an OOV word for typos.
    """
    nbest = min(2, len(oov_word))
    segmentations = [morph_model.viterbi_nbest(oov_word, nbest)[i][0]
                     for i in range(nbest)]
    segmentations = [segmented_word for segmented_word in segmentations if
                     len(segmented_word) > 1]
    
    # If no segmentations with more than 1 segment, it is not possible to get
    # a 2gram or 3gram probability => return False (= no typos found)
    if not segmentations:
        return False
    for segmented_word in segmentations:
        score = lm_segmented.logscore(segmented_word[-1], segmented_word[:-1])
        
        # If any segmentation provides score higher than the threshold, the word
        # is likely to be written correctly
        if score > threshold:
            return False
        
    # All scores were below the threshold => a typo(s) is/are likely to be in 
    # the word
    return True
    
def similar_words(word, morph_model, lm, lm_segmented):
    """
    Search for similar words from the LM vocabulary
    """
    similar_words = []
    nbest = min(5, len(word))
    segmentations = [morph_model.viterbi_nbest(word, nbest)[i][0]
                     for i in range(nbest)]
    for segmented_word in segmentations:
        word_to_search = ''.join(segmented_word[:-1]) #word without last segment
        #word_to_search = max(segmented_word, key=len) #the longest segment 
                        #(does not work for Finnish as well as for Swedish)
        if word_to_search in lm.vocab:
            if word_to_search not in similar_words:
                similar_words.append(word_to_search)
        
        # If the word without its last morpheme is not found from the
        # vocabulary, search for possible word continuations using the 
        # the language model trained on word morphemes
        else:
            possible_continuations = [next_morph[0] for next_morph in
                                      lm_segmented.counts.__getitem__
                                      (segmented_word[:-1]).most_common(3)]
            for morph in possible_continuations:
                similar_word = word_to_search + morph
                if similar_word not in similar_words:
                    if similar_word in lm.vocab:
                        similar_words.append(similar_word)
    return similar_words

def similar_ngrams(ngram, morph_model, lm, nbest):
    """
    Search for similar ngrams from the LM
    """
    w1 = ngram[0]
    w2 = ngram[1]
    nbest_max = 3 # Limit for number of segmentations/word
    sim_ngrams = []
    
    # Preserve only the longest segment of each segmentation
    viterbi_nbest_1 = morph_model.viterbi_nbest(w1, nbest)
    viterbi_nbest_2 = morph_model.viterbi_nbest(w2, nbest)
    segmentations1 = [max([viterbi_nbest_1[i][0] 
                           for i in range(len(viterbi_nbest_1))][j], key=len)
                      for j in range(len(viterbi_nbest_1))]
    segmentations2 = [max([viterbi_nbest_2[i][0] 
                           for i in range(len(viterbi_nbest_2))][j], key=len) 
                      for j in range(len(viterbi_nbest_2))]
    
    # Remove duplicates and too short word segments
    segmentations1 = [segment for segment in list(dict.fromkeys(segmentations1)) 
                      if len(segment)>=min(3,len(w1))]
    segmentations2 = [segment for segment in list(dict.fromkeys(segmentations2)) 
                      if len(segment)>=min(3,len(w2))]
    
    #print(segmentations1)
    #print(segmentations2)
    
    # Put the first word of the bigram itself (w1) to the beginning of the list
    # of words similar to w1.
    similar_words_1 = [w1]
    similar_words_2 = []
    
    # Collect the words from the LM vocabulary that contain
    # the same segment as one of the words of the bigram
    for word in lm.vocab:
        for word1 in segmentations1:
            if word1 in word:
                similar_words_1.append(word)
        for word2 in segmentations2:
            if word2 in word:
                similar_words_2.append(word)
    
    # Make ngrams using all possible combinations of the
    # words found from the LM vocabulary and return the ones
    # that have non-zero probability. Because w1 itself is in the beginning of
    # the list of words similar to w1, bigrams with non-zero probability
    # [w1, _ ] will be returned first (if such bigrams exist)
    for pair in list(itertools.product(similar_words_1, similar_words_2)):
        logscore = lm.logscore(pair[1], [pair[0]])
        if logscore > -float('inf'):
            sim_ngrams.append(pair[0] + ' ' + pair[1])# + ' ' + str(round(logscore)))
        if len(sim_ngrams) > 4:
            break
    
    # If no similar ngrams, try one more time with more word segmentations
    if not sim_ngrams and nbest<nbest_max:
        nbest += 1
        sim_ngrams = similar_ngrams(ngram, morph_model, lm, nbest)
    
    # Remove duplicates
    sim_ngrams = list(dict.fromkeys(sim_ngrams))
    return sim_ngrams

def extract_feats(word, nlp):
    """
    Extract morphological features of a single word
    """
    feat_dict = {}
    feat_string = ''
    doc = nlp(word).to_dict()[0][0]
    if 'feats' in doc:
        for pair in doc['feats'].split('|'):
            feat, val = pair.split('=')
            feat_dict[feat] = val
            feat_string += feat + ': ' + val + ', '
    if feat_string:
        feat_string = ' (' + feat_string[:-2] + ')'
    return feat_dict, feat_string
    
def morph_features(prev_word, pos, nlp, lm):
    """
    Define morphological features of the next word based on the previous word
    """
    feat_string = ''
    feat_dict = {}
    # Collect 15 words which are most likely to follow the previous word w_(i-1)
    # according to the LM and preserve only the ones that belong to the same
    # part of speech as the current word w_i
    for word, _ in lm.counts.__getitem__([prev_word]).most_common(20):
        doc = nlp(word).to_dict()[0][0]
        if 'feats' in doc and (doc['upos'] == pos or doc['xpos'].split('|')[0] == pos):
            for pair in doc['feats'].split('|'):
                feat, val = pair.split('=')
                if feat in feat_dict:
                    feat_dict[feat].append(val)
                else:
                    feat_dict[feat] = [val]
    #print(feat_dict)
    # Find the most common value for each feature
    for dict_key in feat_dict.keys():
        feat_string += dict_key + ': ' + max(set(feat_dict[dict_key]), 
                                        key = feat_dict[dict_key].count) + ', '
    if feat_string:
        feat_string = ' (' + feat_string[:-2] + ')'
    return feat_dict, feat_string

def compare_feats(feat_dict_to_check, feat_dict_correct):
    """
    Check if morphological features of a word can apply to any other words that
    can replace it. Return a string of features with most common values 
    """
    corrected_feats_string = ''
    for dict_key in feat_dict_correct.keys():
        if dict_key not in feat_dict_to_check or feat_dict_to_check[dict_key] \
            not in feat_dict_correct[dict_key]:
                corrected_feats_string += dict_key + ': ' + \
                    max(set(feat_dict_correct[dict_key]), 
                        key = feat_dict_correct[dict_key].count) + ', '
    if corrected_feats_string:
        corrected_feats_string = corrected_feats_string[:-2]
    #print(corrected_feats_string)
    return corrected_feats_string
            
def eval(text_to_analyze, lm, pos_lm, pos_tagger, morph_model, lm_segmented,
         nlp, pos_descr_dict, threshold=float('-inf')):
    
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
    # Search for similar words from the LM vocabulary
    # Also, predict the most likely POS tag t_i for the unkwown word w_i based 
    # on the sequence of previous tags t_(i-2), t_(i-1) using the POS LM
    # In addition to the POS tag, suggest also most likely values for 
    # morphological features
    errs_unk = ""
    err_unk_count = 0
    unk_mark = '<UNK>'
    for sent_idx, sent in enumerate(text):
        for word_idx, word in enumerate(sent):
            if word not in lm.vocab:
                text[sent_idx][word_idx] = word+'*'+unk_mark
                err_unk_count += 1
                errs_unk += str(err_unk_count) + '. ' + word
                
                sim_words = similar_words(word, morph_model, lm, lm_segmented)
                if similar_words:
                    errs_unk += '. Similar words: ' + ', '.join(sim_words) \
                        + '. You can also try to replace the word with some '
                else:
                    errs_unk += '. No similar words found from the vocabulary, ' \
                        + 'try to replace with some '
                    
                next_tag_dict = pos_lm.counts.__getitem__(tags[sent_idx]
                                                           [word_idx-2+(n_pos-1):
                                                            word_idx+(n_pos-1)])
                if len(next_tag_dict)>0:
                    tag_to_use = next_tag_dict.max()
                else:
                    # If no corresponding POS trigrams found from the
                    # POS LM (no tag t_i can follow the tag sequence
                    # t_(i-2), t_(i-1)), use bigrams (search for the most
                    # likely tag t_i given the previous tag t_(i-1))
                    next_tag_dict = pos_lm.counts.__getitem__([tags[sent_idx]
                                                           [word_idx-1+(n_pos-1)]])
                    tag_to_use = next_tag_dict.max()
                
                _, morph_feat_string = morph_features(text[sent_idx][word_idx-1],
                                                      tag_to_use, nlp, lm)
                errs_unk += pos_descr_dict[tag_to_use].lower() + \
                    morph_feat_string + '.\n'
    
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
                    
                    """ 
                    # Suggest 3 most likely words to replace the last word of 
                    # the unknown ngram based on its previous word(s)
                    next_word = ', '.join([pair[0] for pair in 
                                 Counter(lm.counts.__getitem__(ngram[:-1])).most_common(3)])
                    errs += str(err_count) + '. ' + \
                         ' '.join(text[sent_idx][i:i+n]) + ': try to replace the word '\
                             +ngram[-1] + ' with some other word like ' + next_word + '.\n'
                    """
                      
                    errs += str(err_count) + '. ' + ' '.join(ngram)
                    
                    # Search for similar ngrams and suggest at most 5 examples
                    sim_ngrams = similar_ngrams(ngram, morph_model, lm, 1)
                    if sim_ngrams:
                        errs += '. Similar ngrams: ' + ', '.join(sim_ngrams) + '. ' #+ \
                            #'. You can also '
                    else:
                        errs += '. No similar ngrams found from the language model. '
                    
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
                    # part of speech. In this case, check also the morphological
                    # features of the last word of the sequence; ask to pay 
                    # attention to those features for which the current value
                    # is most probably incorrect.
                    # If the log probability is below the average, ask to use
                    # another part of speech and suggest the most likely POS tag
                    # with most likely set of morphological features.
                    pos_logscore = pos_lm.logscore(tags[sent_idx][i +(n_pos-1)],
                                                   tags[sent_idx][i-2+(n_pos-1):
                                                         i +(n_pos-1)])
                    avg_pos_logscore = mean_logprob(tags[sent_idx][i-2+(n_pos-1):
                                                                       i +(n_pos-1)],
                                                        pos_lm)
                    feat_dict_to_check, morph_feat_string = extract_feats(ngram[-1], nlp)
                    #feat_dict_correct, feat_string = morph_features(text[sent_idx][i-1],
                    #                                                tag_to_use,
                    #                                                nlp, lm)
                    errs += 'You used the ' + pos_descr_dict[tags[sent_idx]
                                                             [i +(n_pos-1)]].lower() + ' ' + \
                        ngram[-1] + morph_feat_string + '. '
                    if pos_logscore == -float('inf') or pos_logscore<avg_pos_logscore:
                        #errs += 'You used the ' + pos_descr_dict[tags[sent_idx]
                        #                                         [i +(n_pos-1)]].lower() + ' ' + \
                        #    ngram[-1]
                        next_tag_dict = pos_lm.counts.__getitem__(tags[sent_idx]
                                                           [i-2+(n_pos-1):
                                                            i+(n_pos-1)])
                        if len(next_tag_dict) > 0:
                            tag_to_use = next_tag_dict.max()
                            feat_dict_correct, feat_string = morph_features(text[sent_idx][i],
                                                            tag_to_use, nlp, lm)
                            errs += 'Try to use another part of speech, for example ' + \
                                pos_descr_dict[tag_to_use].lower() + \
                                    feat_string + '.\n'
                            
                        else:
                            # If no corresponding POS trigrams found from the
                            # POS LM (no tag t_i can follow the tag sequence
                            # t_(i-2), t_(i-1)), use bigrams (search for the most
                            # likely tag t_i given the previous tag t_(i-1))
                            pos_logscore = pos_lm.logscore(tags[sent_idx][i +(n_pos-1)],
                                                   [tags[sent_idx][i-1+(n_pos-1)]])
                            avg_pos_logscore = mean_logprob([tags[sent_idx][i-1+(n_pos-1)]], pos_lm)
                            if pos_logscore == -float('inf') or pos_logscore<avg_pos_logscore:
                                next_tag_dict = pos_lm.counts.__getitem__([tags[sent_idx]
                                                            [i-1+(n_pos-1)]])
                                tag_to_use = next_tag_dict.max()
                                feat_dict_correct, feat_string = morph_features(text[sent_idx][i],
                                                                tag_to_use, nlp, lm)
                                errs += 'Try to use another part of speech, for example ' + \
                                    pos_descr_dict[tag_to_use].lower() + \
                                        feat_string + '.\n'
                            else:
                                errs += 'You can also try to use some other ' + pos_descr_dict[tags[sent_idx]
                                                                            [i +(n_pos-1)]].lower() + \
                                ' instead of ' + ngram[-1] + '. '
                                feat_dict_correct, feat_string = morph_features(text[sent_idx][i],
                                                                      tags[sent_idx][i+(n_pos-1)],
                                                                      nlp, lm)
                                corr_feat_string = compare_feats(feat_dict_to_check,
                                                                 feat_dict_correct)
                                if corr_feat_string:
                                    errs += 'It is also recommended to correct' \
                                        ' the following morphological features: ' \
                                            + corr_feat_string + '.'
                                errs += '\n'
                    else:      
                        errs += 'You can also try to use some other ' + pos_descr_dict[tags[sent_idx]
                                                                          [i +(n_pos-1)]].lower() + \
                            ' instead of ' + ngram[-1] + '. '
                        feat_dict_correct, feat_string = morph_features(text[sent_idx][i],
                                                              tags[sent_idx][i+(n_pos-1)], nlp, lm)
                        corr_feat_string = compare_feats(feat_dict_to_check,
                                                         feat_dict_correct)
                        if corr_feat_string:
                            errs += 'It is also recommended to correct the' \
                                ' following morphological featutes: ' \
                                    + corr_feat_string + '.'
                                
                        errs += '\n'
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
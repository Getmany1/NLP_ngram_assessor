import morfessor
import os

def train_morfessor(corpus, split_prob):
    """
    Train Morfessor Baseline model
    Lowercase the input text; use random skips for frequently seen compounds
    to speed up training; initialize new words by random splitting using the 
    split probability of split_prob.
    """
    
    io = morfessor.MorfessorIO(compound_separator=r"[^-\w]+" ,lowercase=True)
    
    train_data = list(io.read_corpus_file(os.path.join('data','corpora', corpus)))

    model_tokens = morfessor.BaselineModel(use_skips=True)

    model_tokens.load_data(train_data, init_rand_split=split_prob)

    model_tokens.train_batch()
    
    io.write_binary_model_file(os.path.join('data','models', corpus[:-4] + '_morph'), 
                               model_tokens)
    
    return model_tokens
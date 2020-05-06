# NLP_ngram_assessor
A tool for evaluating sentences using an N-gram language model. The model can be trained using a text corpus. Alternatively, a pretrained model can be loaded. The evaluating script checks the sentences for out-of-vocabulary (OOV) words and highligh them with \<UNK> mark. In addition, it highlights uncommon N-grams. N-grams which have log-probability below the predefined theshold are supposed to be uncommon. By default, the threshold is set to -infinity, which corresponds to a log-probability of an unseen N-gram. 
  
  ## Example
  ### Example text to evaluate
  "This is a test text. The automatic assessor will report OOV words and uncommon ngrams."
  
  ### Script output:
  
Text to evaluate:
This is a test text. The automatic assessor will report OOV words and uncommon ngrams.

Analyzed text with annotations:
\<s> this is a test text . \</s> \<s> the automatic assessor\*1 will\*2 report oov*\<UNK> words and uncommon ngrams*\<UNK> . \</s>

Uncommon ngrams:
1. automatic assessor
2. assessor will

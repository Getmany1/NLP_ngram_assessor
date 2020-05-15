# NLP_ngram_assessor
A tool for evaluating sentences using an N-gram language model. The model can be trained using a text corpus. Alternatively, a pretrained model can be loaded. The evaluating script checks the sentences for out-of-vocabulary (OOV) words and highligh them with \<UNK> mark. In addition, it highlights uncommon N-grams. N-grams which have log-probability below the predefined theshold are supposed to be uncommon. By default, the threshold is set to -infinity, which corresponds to a log-probability of an unseen N-gram. 
  
  ## Example
  ### Example text to evaluate
  "This is a test text. The automatic assessor will report OOV words and uncommon ngrams."
  
  ### Script output:
  
Text to evaluate: 
This is a test text. The automatic assessor will report OOV words and uncommon what ngrams.

(For evaluator) Text tagged with part-of-speech tags: 
[[('This', 'DT'), ('is', 'VBZ'), ('a', 'DT'), ('test', 'NN'), ('text', 'NN'), ('.', '.')], [('The', 'DT'), ('automatic', 'JJ'), ('assessor', 'NN'), ('will', 'MD'), ('report', 'VB'), ('OOV', 'NNP'), ('words', 'NNS'), ('and', 'CC'), ('uncommon', 'JJ'), ('what', 'WP'), ('ngrams', 'NNS'), ('.', '.')]]

Analyzed text with annotations: 
this is a test text . the automatic assessor\*1 will\*2 report oov*\<UNK> words and uncommon  what\*3 ngrams*\<UNK> .

Unknown words: 
1. oov: try to replace with some determiner.
2. ngrams: try to replace with some verb (past tense).

Uncommon ngrams: 
1. automatic assessor: try to use some other noun (singular or mass) instead of assessor.
2. assessor will: try to use some other modal instead of will.
3. uncommon what: you used the wh-pronoun what; try to use another part of speech, for example noun (singular or mass).

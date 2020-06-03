# NLP_ngram_assessor
A tool for evaluating sentences using an N-gram language model. The model can be trained using a text corpus. Alternatively, a pretrained model can be loaded. The evaluating script checks the sentences for out-of-vocabulary (OOV) words and highligh them with \<UNK> mark. In addition, it highlights uncommon N-grams. N-grams which have log-probability below the predefined theshold are supposed to be uncommon. By default, the threshold is set to -infinity, which corresponds to a log-probability of an unseen N-gram.

## Link for data (corpora & models) for Aalto users
https://drive.google.com/open?id=1vh1SMU1g5Wd7xsRH8UoHVlOk50K9vSqH
  
  ## Example
  ### Example text to evaluate
  "Oavsett regleringen i gymnasieskolans styrdokument att beakta samtliga kommunikativa delfärdigheter, saknas det muntliga testet
  fortfarande i den finländska studentexamen."
  
  ### Script output:
  
Text to evaluate: 
Oavsett regleringen i gymnasieskolans styrdokument att beakta samtliga kommunikativa delfärdigheter, saknas det muntliga testet fortfarande i den finländska studentexamen.

(For human evaluator) Text tagged with part-of-speech tags: 
[[('Oavsett', 'ADJ'), ('regleringen', 'NOUN'), ('i', 'ADP'), ('gymnasieskolans', 'NOUN'), ('styrdokument', 'NOUN'), ('att', 'PART'), ('beakta', 'VERB'), ('samtliga', 'ADJ'), ('kommunikativa', 'ADJ'), ('delfärdigheter', 'NOUN'), (',', 'PUNCT'), ('saknas', 'VERB'), ('det', 'DET'), ('muntliga', 'ADJ'), ('testet', 'NOUN'), ('fortfarande', 'ADV'), ('i', 'ADP'), ('den', 'DET'), ('finländska', 'ADJ'), ('studentexamen', 'NOUN'), ('.', 'PUNCT')]]

Analyzed text with annotations: 
oavsett regleringen\*1 i gymnasieskolans\*<UNK> styrdokument att\*2 beakta samtliga kommunikativa\*3 delfärdigheter\*<UNK> , saknas det muntliga testet\*4 fortfarande\*5 i den finländska studentexamen .

Unknown words: 
1. gymnasieskolans. Similar words: gymnasie, gymnasieskola. You can also try to replace the word with some noun.
2. delfärdigheter. Similar words: färdigheter, färdig, färdighet. You can also try to replace the word with some noun.

Uncommon ngrams: 
1. oavsett regleringen. Similar ngrams: sett reglering. You used the noun regleringen. You can also try to use some other noun instead of regleringen.
2. styrdokument att. Similar ngrams: dokumentärfilmen att, dokumenten att, dokumentera att, dokumentär att, dokument att. You used the particle att. Try to use another part of speech, for example adposition.
3. samtliga kommunikativa. Similar ngrams: samt kommunikativa. You can also try to use some other adjective instead of kommunikativa.
4. muntliga testet. Similar ngrams: egentliga testerna, offentliga protesterna, offentliga missiltestet, ordentliga test. You can also try to use some other noun instead of testet.
5. testet fortfarande. Similar ngrams: protesterar fortfarande, testas fortfarande, testerna fortfarande. You can also try to use some other adverb instead of fortfarande.

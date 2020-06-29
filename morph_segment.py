import dill as pickle
import morfessor
import os

corpus = 'Wikipedia_fi_2017.pkl'
io = morfessor.MorfessorIO(compound_separator=r"[^-\w]+" ,lowercase=True)
morph_model = io.read_binary_model_file(os.path.join('data','models','Wikipedia_fi_2017_morph'))
text_segmented = []

# Read the corpus file
if corpus[-4:] == '.txt':
    with open(os.path.join('data','corpora',corpus), encoding='utf8') as f:
        text = f.read()

elif corpus[-4:] == '.pkl':
    with open(os.path.join('data','corpora',corpus), 'rb') as f:
        text = pickle.load(f)

if type(text) is str:
    text = text.lower()
    text = sent_tokenize(text)
    text = [word_tokenize(sent) for sent in text]

for sent_idx, sent in enumerate(text):
    sent_segmented = []
    for word_idx, word in enumerate(sent):
        [sent_segmented.append(segment) for segment in
         morph_model.viterbi_nbest(word.lower(),1)[0][0]]
    text_segmented.append(sent_segmented)
    
print(len(text_segmented))
# Save the corpus
print('saving...')
with open(os.path.join('data','corpora', corpus[-4:] + '_morph_segmented.pkl'), 'wb') as f:
    pickle.dump(text_segmented, f, 4)
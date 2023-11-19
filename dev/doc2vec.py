# Riorganizzazione del codice del chap. 12 relativo a Doc2Vec
"""
1. text8 va scaricato manualmente da http://mattmahoney.net/dc/text8.zip
2. Fare riferimento al file word2vec.py
3. Necessario `pip install testfixtures`
"""

## Imports
import gensim
import smart_open
from gensim.models import doc2vec
from gensim.models import Doc2Vec
import os
import site
site_pkgs = site.getsitepackages()[0]
gs_path = gensim.__path__[0]

# Path dove risiede text8
CORPUSDATA = os.path.expanduser('~') + '/studio/data/corpus'
# Gensim Data Dir
test_data = os.path.join(gensim.__path__[0], 'test', 'test_data')

## Load corpus
lee_train_file = os.path.join(test_data, 'lee_background.cor')
lee_test_file = os.path.join(test_data, 'lee.cor')

def read_corpus(file_name, tokens_only=False):
    """
    Read corpus
    Attenzione: costruisce anche la parte di labeling necessaria per Doc2Vec
    """
    with smart_open.smart_open(file_name) as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

train_corpus = list(read_corpus(lee_train_file))
test_corpus = list(read_corpus(lee_test_file, tokens_only=True))

## Build Vocabulary and Train Model
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=100)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

## Costruisce i modelli separati (PV-DBOW e PV-DM) e li concatena
models = [
    # PV-DBOW
    Doc2Vec(dm=0, dbow_words=1, vector_size=200, window=8, min_count=10, epochs=50),

    # PV-DM w/average
    Doc2Vec(dm=1, dm_mean=1, vector_size=200, window=8, min_count=10, epochs=50),
]
documents = train_corpus

models[0].build_vocab(documents)
models[1].reset_from(models[0])

for model in models:
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

from gensim.test.test_doc2vec import ConcatenatedDoc2Vec

new_model = ConcatenatedDoc2Vec((models[0], models[1]))

## Check Results
inferred_vector = model.infer_vector(train_corpus[0].words)
sims = model.dv.most_similar([inferred_vector])
print(sims)
# Riorganizzazione del codice del chap. 12
"""
1. text8 va scaricato manualmente da http://mattmahoney.net/dc/text8.zip
"""
## Imports
import gensim
from gensim.models import word2vec
import os
import site
site_pkgs = site.getsitepackages()[0]
gs_path = gensim.__path__[0]

# Path dove risiede text8
CORPUSDATA = os.path.expanduser('~') + '/studio/data/corpus'

## Data Setup

sentences = word2vec.Text8Corpus(CORPUSDATA + '/text8')
model = word2vec.Word2Vec(sentences, vector_size=200, hs=1)

print(model)

## Analisi

model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)[0]

## Most Similar Cosmul
model.wv.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])

## Restituisce il vettore di 'computer'
model.wv['computer']

## Altri esempi
x = model.wv.doesnt_match("breakfast cereal dinner lunch".split())
print(x)

x = model.wv.similarity('woman', 'man')
print(x)

x = model.wv.similarity('woman', 'cereal')
print(x)

x = model.wv.distance('man', 'woman')
print(x)

## Esempio di Training
import site
site_pkgs = site.getsitepackages()[0]
test_data = os.path.join(gs_path, '/test/test_data')

##
model.wv.evaluate_word_pairs(os.path.join(test_data, 'wordsim353.tsv'))

## Valutazione accuratezza (modificata perché deprecata)
score, predictions = model.wv.evaluate_word_analogies(os.path.join(test_data, 'questions-words.txt'))
# model.wv.accuracy(os.path.join(test_data, 'questions-words.txt'))
print(score)
print(predictions)



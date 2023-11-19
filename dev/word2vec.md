# word2vec
Si tratta del codice del capitolo 12 riorganizzato per funzionare correttamente.
1. Va fissato il path dei dati, all'inizio del codice, dove risiede il corpus, nel nostro caso `text8`;
2. Modificato il parametro `size` in `vector_size` (non più presente)
3. Cambiato il codice relativo ad `accuracy` (ora deprecato) con `evaluate_word_analogies`
4. È necessario effettuare il download di `text8` manualmente (nel path di cui al primo punto).
5. Gli altri file fanno parte di gensim


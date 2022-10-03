##tensorflow >2.0
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
import numpy as np
### sentences
### sentences
sent=[  'boy is good', 'the glass of milk' ]
### Vocabulary size
voc_size=10000
#one hot
onehot_repr=[one_hot(words,voc_size)for words in sent] 
#padding
sent_length=8
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
dim=10
model=Sequential()
model.add(Embedding(voc_size,10,input_length=sent_length))
model.compile('adam','mse')
model.summary()
all = model.predict(embedded_docs)
oneHot = embedded_docs[0]
wordEmbedding = model.predict(embedded_docs)[0]

#Dataset: https://www.kaggle.com/c/fake-news/data#
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import re
import numpy as np
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

df=pd.read_csv('fake-news/train.csv')
df=df.dropna()

messages=df.copy()
messages.reset_index(inplace=True)

ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()   
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
y=messages['label']

### Vocabulary size
voc_size=5000
#one hot
onehot_repr=[one_hot(words,voc_size)for words in corpus] 
#padding
sent_length=20
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)

## Creating model
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(100)))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())
all = model.predict(embedded_docs)

X_final=np.array(embedded_docs)
y_final=np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)
### Finally Training
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)


y_pred = model.predict(X_test)
y_pred = np.round(y_pred).astype(int)

from sklearn.metrics import confusion_matrix
c = confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score
a = accuracy_score(y_test,y_pred)





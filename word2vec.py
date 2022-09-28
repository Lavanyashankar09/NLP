import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from matplotlib import pyplot
import re

paragraph = """	Have a great day
    			Have a good day
			    Have a nice day
                """

# Preprocessing the data
text = re.sub(r'\[[0-9]*\]',' ',paragraph)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)

# Preparing the dataset
sentences = nltk.sent_tokenize(text)
sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]  
    
# Training the Word2Vec model
model = Word2Vec(sentences, min_count=1, vector_size=100)
# Finding the vocab
vocab1 = model.wv.key_to_index
# Finding Word Vectors
vector = model.wv['good']
# Most similar words
similar = model.wv.most_similar('nice')

#for graph
vocab = list(model.wv.key_to_index)
X = model.wv[vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)

# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.key_to_index)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()

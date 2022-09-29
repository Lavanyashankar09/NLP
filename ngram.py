# with nltk
import re
from nltk.util import ngrams
s = "the quick brown fox "
tokens = [token for token in s.split(" ") if token != ""]
output = list(ngrams(tokens, 1))

#with vectorizer
c = ["the quick brown fox", "jump over lazy dog"]
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(1,1))
X = cv.fit_transform(c).toarray()
feature_names = cv.get_feature_names_out()
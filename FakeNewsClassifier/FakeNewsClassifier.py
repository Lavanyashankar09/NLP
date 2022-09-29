#Dataset: https://www.kaggle.com/c/fake-news/data#
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
df=pd.read_csv('fake-news/train.csv')

df=df.dropna()
messages=df.copy()
messages.reset_index(inplace=True)

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()   
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
########################################################    
## Applying Countvectorizer
# Creating the Bag of Words model
cv = CountVectorizer(max_features=5000,ngram_range=(1,3))
X = cv.fit_transform(corpus).toarray()
feature_names = cv.get_feature_names()

############################################################

hs_vectorizer=HashingVectorizer(n_features=5000)
X=hs_vectorizer.fit_transform(corpus).toarray()

############################################################

y=messages['label']
## Divide the dataset into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

#############################################################
# Training model using Naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
model1 = MultinomialNB().fit(X_train, y_train)
y_pred=model1.predict(X_test)

con_mat = confusion_matrix(y_test, y_pred)
accu = accuracy_score(y_test, y_pred)

################################################################
# Training model using Passive Aggressive classifier
from sklearn.linear_model import PassiveAggressiveClassifier
linear_clf = PassiveAggressiveClassifier(max_iter=50)
model1 = linear_clf.fit(X_train, y_train)
y_pred=model1.predict(X_test)

con_mat = confusion_matrix(y_test, y_pred)
accu = accuracy_score(y_test, y_pred)

##############################################################
#Multinomial Classifier with Hyperparameter
from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB(alpha=0.1)
previous_score=0
for alpha in np.arange(0,1,0.1):
    sub_classifier=MultinomialNB(alpha=alpha)
    sub_classifier.fit(X_train,y_train)
    y_pred=sub_classifier.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    if score>previous_score:
        classifier=sub_classifier
    print("Alpha: {}, Score : {}".format(alpha,score))
    
###############################################################
from sklearn.linear_model import LinearRegression
model1 = LinearRegression().fit(X_train, y_train)
y_pred=model1.predict(X_test)

con_mat = confusion_matrix(y_test, y_pred)
accu = accuracy_score(y_test, y_pred)

# Get Features names
feature_names = cv.get_feature_names()
model1.coef_[0]
### Most real
sorted(zip(model1.coef_[0], feature_names), reverse=True)[:20]
### Most fake
sorted(zip(model1.coef_[0], feature_names))[:5000]
    
    
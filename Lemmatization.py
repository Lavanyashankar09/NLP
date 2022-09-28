import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

paragraph = """I have three visions for India. In 3000 years of our history. goes. finally"""           
               
sentences = nltk.sent_tokenize(paragraph)
lemmatizer = WordNetLemmatizer()

# Lemmatization
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)      
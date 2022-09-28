

# Tokenization of paragraphs/sentences
import nltk
#nltk.download()

paragraph = """I have three visions for India. In 3000 years of our history,conquered our minds. 
               """
               
# Tokenizing sentences
sentences = nltk.sent_tokenize(paragraph)

# Tokenizing words
words = nltk.word_tokenize(paragraph)






import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import string
import re
import os
import codecs
from sklearn import feature_extraction

import gensim

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

##################################
#####Data preprocessing###########

#Tokenization function
def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


###################################
###########Predicting topic#######

pred_doc ='boots,woodland'
print pred_doc
tokenized_text = [tokenize_only(pred_doc)]
text = [[word for word in text if word not in stopwords] for text in tokenized_text]
dictionary = corpora.Dictionary.load('predict.dict')
pred_vec = [dictionary.doc2bow(i) for i in text]
print pred_vec
print dictionary
lda = gensim.models.LdaModel.load('lda_ads_model.lda')

for top in lda.print_topics():
	print top

new_topics = lda[pred_vec]

for topic in new_topics:
    print(topic)

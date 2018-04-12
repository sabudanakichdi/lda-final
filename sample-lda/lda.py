
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim


tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')


# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
    

##################################

#######Importing dataset #########

import os.path
raw_documents = []
snippets = []
with open( os.path.join("dataset", "train.csv") ,"r") as fin:
    for line in fin.readlines():
        text = line.strip()
        raw_documents.append( text )
        # keep a short snippet of up to 100 characters as a title for each article
        snippets.append( text[0:min(len(text),100)] )
print("Read %d raw text documents" % len(raw_documents))

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

#Stop words and tokenize
nltk.download('punkt')
tokenized_text = [tokenize_only(text) for text in raw_documents]
texts = [[word for word in text if word not in stopwords] for text in tokenized_text]

###################################
#####LDA Model Trainig ############

#turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
#remove extremes (similar to the min/max df step used when creating the tf-idf matrix)
dictionary.filter_extremes(no_below=1, no_above=0.8)
#saving dict for prediction
dictionary.save('predict.dict') 

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(i) for i in texts]

###########Generate LDA model#######
lda = gensim.models.ldamodel.LdaModel(corpus, num_topics=20, id2word = dictionary, update_every=5, chunksize=10000, passes=50, minimum_probability=0)
lda.save('lda_ads_model.lda')


for top in lda.print_topics():
	print top


###########Predicting topic#######
pred_doc ='nike,running,shoes'

tokenized_text = [tokenize_only(pred_doc)]
text = [[word for word in text if word not in stopwords] for text in tokenized_text]
dictionary = corpora.Dictionary.load('predict.dict')
pred_vec = [dictionary.doc2bow(i) for i in text]
print pred_vec
print dictionary
#new_lda = gensim.models.LdaModel.load('lda_ads_model.lda')
new_topics = lda[pred_vec]

for topic in new_topics:
    print(topic)



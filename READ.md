# Targeted Advertising using Behavioral Data Mining

This project is a service that, based on user behavior in browser, generates relevant advertisement using topic based user segmentation LDA algorithm and XGBoost classifier.

# LDA (Latent Dirichlet allocation):

In natural language processing, latent Dirichlet allocation (LDA) is a generative statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar. For example, if observations are words collected into documents, it posits that each document is a mixture of a small number of topics and that each word's presence is attributable to one of the document's topics. 
LDA is an example of a topic model. Its an unsupervised topic modelling (has better performance than K-means in this case)

# Why LDA over K-means: 

REF: https://www.researchgate.net/publication/220267053_Topic-Based_User_Segmentation_for_Online_Advertising_with_Latent_Dirichlet_Allocation

REF of LDA: https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/

REF of LDA2VEC: https://github.com/cemoody/lda2vec

# XgBoost:

Supervised Classifier
XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework.

# Technology Used:

1. AWS AMI Deep Learning. (Server)
2. Gensim (Library for LDA)
3. XgBoost (Library for Classifier)

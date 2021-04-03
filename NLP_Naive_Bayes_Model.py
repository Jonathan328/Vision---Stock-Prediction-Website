import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import sent_tokenize,word_tokenize
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize
from gensim.parsing.preprocessing import remove_stopwords

 
short_pos = open('Finnews.positive.txt').read() 
short_neg = open('Finnews.negative.txt').read()

#Preprocess Financial news training data

nostopwords_pos = remove_stopwords(short_pos)
nostopwords_neg = remove_stopwords(short_neg)

tokenized_p = word_tokenize(nostopwords_pos)
sent2_p = nltk.pos_tag(tokenized_p)
sent3_p = nltk.ne_chunk(sent2_p)

tokenclean_p = tokenized_p[:]

for i in range(len(sent3_p)):
    if len(sent3_p[i])== 1:
        tokenclean_p.remove(sent3_p[i][0][0])

tokenclean_p = ' '.join(tokenclean_p)


tokenized_n = word_tokenize(nostopwords_neg)
sent2_n = nltk.pos_tag(tokenized_n)
sent3_n = nltk.ne_chunk(sent2_n)

tokenclean_n = tokenized_n[:]

for i in range(len(sent3_n)):
    if len(sent3_n[i])== 1:
        tokenclean_n.remove(sent3_n[i][0][0])
tokenclean_n = ' '.join(tokenclean_n)

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

allwords = []
documents = []
allowed_word_types = ['J','V'] # Part of speech of words in training data: J=Adj, R=Adverb, V=Verb (Only Adjective and Verb is allowed) 

for r in tokenclean_p.split('.'):
    documents.append((r,'pos'))
    words = word_tokenize(r)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            allwords.append(w[0].lower())

for r in tokenclean_n.split('.'):
    documents.append((r,'neg'))
    words = word_tokenize(r)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            allwords.append(w[0].lower())


save_documents = open('documents.pickle','wb')
pickle.dump(documents,save_documents)
save_documents.close()



#Most Common words in all the text doucments (Including punctuations)
all_words = nltk.FreqDist(allwords)


#Top 3000 words
word_feature = list(all_words.most_common(5000))
word_features = []
for i in word_feature:
    word_features.append(i[0])
    

save_wordfeatures = open('wordfeatures.pickle','wb')
pickle.dump(word_features,save_wordfeatures)
save_wordfeatures.close()


#Check if words in the documents are of the top 3000 words
def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    
    return features

#print(find_features(text))

featuresets = [(find_features(rev),category) for (rev,category) in documents]


save_featureset = open('featureset.pickle','wb')
pickle.dump(featuresets,save_featureset)
save_featureset.close()

print(len(featuresets))
random.shuffle(featuresets)

training_set = featuresets[:1700]
testing_set = featuresets[1700:]


#Original Naive Bayes
classifier = nltk.NaiveBayesClassifier.train(training_set)
print('Naive Bayes Algo accuracy percent:', (nltk.classify.accuracy(classifier,testing_set)))
save_classifier = open('naivebayes.pickle','wb')
pickle.dump(classifier, save_classifier)
save_classifier.close()

#MNB classifier 
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print('MNB_classifier accuracy', (nltk.classify.accuracy(MNB_classifier,testing_set)))

save_MNB_classifier = open('MNBclassifier.pickle','wb')
pickle.dump(MNB_classifier, save_MNB_classifier)
save_MNB_classifier.close()


#BernoulliNB_classifier
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print('BernoulliNB_classifier accuracy:',(nltk.classify.accuracy(BernoulliNB_classifier,testing_set)))

save_BernoulliNB_classifier = open('BernoulliNBclassifier.pickle','wb')
pickle.dump(BernoulliNB_classifier,save_BernoulliNB_classifier)
save_BernoulliNB_classifier.close()

#LogisticRegression_classifier
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print('LogisticRegression_classifier accuracy', (nltk.classify.accuracy(LogisticRegression_classifier,testing_set)))

save_LogisticRegression = open('LogisticRegression.pickle','wb')
pickle.dump(LogisticRegression_classifier,save_LogisticRegression)
save_LogisticRegression.close()


#SGDClassifer_classifer
SGDClassifer_classifier = SklearnClassifier(SGDClassifier())
SGDClassifer_classifier.train(training_set)
print('SGDClassifier accuracy', (nltk.classify.accuracy(SGDClassifer_classifier,testing_set)))

save_SGDC = open('SGDC.pickle','wb')
pickle.dump(SGDClassifer_classifier,save_SGDC)
save_SGDC.close()

#SVClassifer_classifier
SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print('SGDClassifer accuracy', (nltk.classify.accuracy(SVC_classifier,testing_set)))

save_SVC = open('SVC.pickle','wb')
pickle.dump(SVC_classifier,save_SVC)
save_SVC.close()

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print('LinearSVC_classifier accuracy', (nltk.classify.accuracy(LinearSVC_classifier, testing_set)))

save_LinearSVC = open('LinearSVC.pickle','wb')
pickle.dump(LinearSVC_classifier,save_LinearSVC)
save_LinearSVC.close()

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print('NuSVC_classifier accuracy', (nltk.classify.accuracy(NuSVC_classifier, testing_set)))

save_NuSVC = open('NuSVC_classifier','wb')
pickle.dump(NuSVC_classifier,save_NuSVC)
save_NuSVC.close()



# VoteClassifier is a class summarize the estimation of all classifiers

class VoteClassifier(ClassifierI):
    
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    
    def classify(self, features):       # classify outputs the sentiment of the sentnece (pos / neg) by determining whether more classifier estimate the sentence to be positive or negative 
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self,features):   # confidence outputs the average confidence level of all classifiers 
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        print(mode(votes))
        conf = choice_votes / len(votes)
        return conf

      
# Testing 
voted_classifier = VoteClassifier(classifier,NuSVC_classifier,LinearSVC_classifier,MNB_classifier,BernoulliNB_classifier,LogisticRegression_classifier,SGDClassifer_classifier,SVC_classifier)                     
print(nltk.classify.accuracy(voted_classifier, testing_set))
print(voted_classifier.classify(testing_set[5][0]), 'Confidence %', voted_classifier.confidence(testing_set[5][0]))
print(voted_classifier.classify(testing_set[1][0]), 'Confidence %', voted_classifier.confidence(testing_set[1][0]))
print(voted_classifier.classify(testing_set[2][0]), 'Confidence %', voted_classifier.confidence(testing_set[2][0]))




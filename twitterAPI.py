# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 09:36:09 2016

@author: Filip
"""

import twitter
import re
import time
import csv
import re
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.corpus import stopwords
import nltk
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.svm import SVC 


corpusFile="C:\Users\..\corpus.csv"
tweetDataFile="C:\Users\..\tweetDataFile.csv"

api = twitter.Api(consumer_key='Enter your key',
                 consumer_secret='Enter your key',
                 access_token_key='Enter your token',
                 access_token_secret='Enter your token')

#print(api.VerifyCredentials())


def createTestData(search_string):
    try:
        tweets_fetched=api.GetSearch(search_string, count=100)
        print "We fetched "+str(len(tweets_fetched))+" tweets with the term "+search_string+"!!"
        return [{"text":status.text,"label":None} for status in tweets_fetched]
    except:
        print "Error!"
        return None
    
search_string=input("Enter search string: ")
testData=createTestData(search_string)

#print testData[0:9]



def createTrainingCorpus(corpusFile,tweetDataFile):
    import csv
    corpus=[]
    with open(corpusFile,'rb') as csvfile:
        lineReader = csv.reader(csvfile,delimiter=',',quotechar="\"")
        for row in lineReader:
            corpus.append({"tweet_id":row[2],"label":row[1],"topic":row[0]})
            
    rate_limit=180
    sleep_time=900/180
    
    trainingData=[]
    for tweet in corpus:
        try:
            status=api.GetStatus(tweet["tweet_id"])
            print "Tweet fetched" + status.text
            tweet["text"]=status.text
            trainingData.append(tweet)
            time.sleep(sleep_time)  
        except: 
            continue
        
    with open(tweetDataFile,'wb') as csvfile:
        linewriter=csv.writer(csvfile,delimiter=',',quotechar="\"")
        for tweet in trainingData:
            try:
                linewriter.writerow([tweet["tweet_id"],tweet["text"],tweet["label"],tweet["topic"]])
            except Exception,e:
                print e
    return trainingData
    
    
    
def createLimitedTrainingCorpus(corpusFile,tweetDataFile):
    import csv
    corpus=[]
    with open(corpusFile,'rb') as csvfile:
        lineReader = csv.reader(csvfile,delimiter=',',quotechar="\"")
        for row in lineReader:
            corpus.append({"tweet_id":row[2],"label":row[1],"topic":row[0]})

    
    
    trainingData=[]
    for label in ["positive","negative"]:
        i=1
        for tweet in corpus:
            if tweet["label"]==label and i<=50:
                try:
                    status=api.GetStatus(tweet["tweet_id"])
                    print "Tweet fetched" + status.text
                    tweet["text"]=status.text
                    trainingData.append(tweet)
                    i=i+1
                except Exception, e: 
                    print e
                    
    with open(tweetDataFile,'wb') as csvfile:
        linewriter=csv.writer(csvfile,delimiter=',',quotechar="\"")
        for tweet in trainingData:
            try:
                linewriter.writerow([tweet["tweet_id"],tweet["text"],tweet["label"],tweet["topic"]])
            except Exception,e:
                print e
    return trainingData
    

#trainingData=createTrainingCorpus(corpusFile,tweetDataFile)
#print trainingData
#trainingData=createLimitedTrainingCorpus(corpusFile,tweetDataFile)


class PreProcessTweets:
    def __init__(self):
        self._stopwords=set(stopwords.words('english')+list(punctuation)+['AT_USER','URL'])
        
    def processTweets(self,list_of_tweets):
        # The list of tweets is a list of dictionaries which should have the keys, "text" and "label"
        processedTweets=[]
        # This list will be a list of tuples. Each tuple is a tweet which is a list of words and its label
        for tweet in list_of_tweets:
            processedTweets.append((self._processTweet(tweet["text"]),tweet["label"]))
        return processedTweets
    
    def _processTweet(self,tweet):
        # 1. Pretvoriti sve u mala slova
        tweet=tweet.lower()
        # 2. zamjena linkova sa stringom URL 
        tweet=re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)     
        # 3. zamjeniti @username sa "AT_USER"
        tweet=re.sub('@[^\s]+','AT_USER',tweet)
        # 4. zamjeniti #word sa word 
        tweet=re.sub(r'#([^\s]+)',r'\1',tweet)
        tweet=word_tokenize(tweet) 
        return [word for word in tweet if word not in self._stopwords]
    
tweetProcessor=PreProcessTweets()
ppTrainingData=tweetProcessor.processTweets(trainingData)
ppTestData=tweetProcessor.processTweets(testData)



############################################################### S V M ###################################################

svmTrainingData=[' '.join(tweet[0]) for tweet in ppTrainingData]

vectorizer=CountVectorizer(min_df=1)
X=vectorizer.fit_transform(svmTrainingData).toarray()
vocabulary=vectorizer.get_feature_names()



swn_weights=[]

for word in vocabulary:
    try:
        synset=list(swn.senti_synsets(word))
        common_meaning =synset[0]
        if common_meaning.pos_score()>common_meaning.neg_score():
            weight=common_meaning.pos_score()
        elif common_meaning.pos_score()<common_meaning.neg_score():
            weight=-common_meaning.neg_score()
        else: 
            weight=0
    except: 
        weight=0
    swn_weights.append(weight)
    
    



swn_X=[]
for row in X: 
    swn_X.append(np.multiply(row,np.array(swn_weights)))
swn_X=np.vstack(swn_X)


#ako koristimo Naive Bayes odkomentirati:
labels_to_array={"positive":1,"negative":2, "neutral":3, "irrelevant":4}
#ako koristimo SVM odkomentirati:
#labels_to_array={"positive":1,"negative":2}
labels=[labels_to_array[tweet[1]] for tweet in ppTrainingData]
y=np.array(labels)


SVMClassifier=SVC()
SVMClassifier.fit(swn_X,y)



# SVM 
SVMResultLabels=[]
for tweet in ppTestData:
    tweet_sentence=' '.join(tweet[0])
    svmFeatures=np.multiply(vectorizer.transform([tweet_sentence]).toarray(),np.array(swn_weights))
    SVMResultLabels.append(SVMClassifier.predict(svmFeatures)[0])





if SVMResultLabels.count(1)>SVMResultLabels.count(2):
    print "SVM Result Positive Sentiment " + str(100*SVMResultLabels.count(1)/len(SVMResultLabels))+" %"
else: 
    print "SVM Result Negative Sentiment " + str(100*SVMResultLabels.count(2)/len(SVMResultLabels))+" %"
    
    

print testData[0:99]
print SVMResultLabels[0:99]




########################################### N A I V E     B A Y E S #################################################################


def buildVocabulary(ppTrainingData):
    all_words=[]
    for (words,sentiment) in ppTrainingData:
        all_words.extend(words)
    wordlist=nltk.FreqDist(all_words)
    # This will create a dictionary with each word and its frequency
    word_features=wordlist.keys()
    # This will return the unique list of words in the corpus 
    return word_features

# NLTK has an apply_features function that takes in a user-defined function to extract features 
# from training data. We want to define our extract features function to take each tweet in 
# The training data and represent it with the presence or absence of a word in the vocabulary 

def extract_features(tweet):
    tweet_words=set(tweet)
    features={}
    for word in word_features:
        features['contains(%s)' % word]=(word in tweet_words)
        # This will give us a dictionary , with keys like 'contains word1' and 'contains word2'
        # and values as True or False 
    return features 

# Now we can extract the features and train the classifier 
word_features = buildVocabulary(ppTrainingData)
trainingFeatures=nltk.classify.apply_features(extract_features,ppTrainingData)
# apply_features will take the extract_features function we defined above, and apply it it 
# each element of ppTrainingData. It automatically identifies that each of those elements 
# is actually a tuple , so it takes the first element of the tuple to be the text and 
# second element to be the label, and applies the function only on the text 

NBayesClassifier=nltk.NaiveBayesClassifier.train(trainingFeatures)


NBResultLabels=[NBayesClassifier.classify(extract_features(tweet[0])) for tweet in ppTestData]


if NBResultLabels.count('positive')>NBResultLabels.count('negative'):
    print "NB Result Positive Sentiment " + str(100*NBResultLabels.count('positive')/len(NBResultLabels))+" %"
else: 
    print "NB Result Negative Sentiment " + str(100*NBResultLabels.count('negative')/len(NBResultLabels))+" %"
    
    


print testData[0:99]
print NBResultLabels[0:99]



#####################################################################################################################





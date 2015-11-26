import sys
import os
import pprint
import json
import re
#import numpy as np


def main():
    tweet_file = open(sys.argv[1])

    hashTagFreqs = {}
    
    def putHashtag(hashtag, hashtagFreqs):
        try:
            junk = hashtag.encode("UTF-8")
        except UnicodeDecodeError:
            hashtag = hashtag.decode("UTF-8")
        
        if (hashtag in hashtagFreqs.keys()): # append to existing list
            hashtagFreqs[hashtag] += 1
            return(hashtagFreqs)
        else: # add key
            hashtagFreqs[hashtag] = 1
            return(hashtagFreqs)

    tweetIx = 0  # used for debugging only

    # compute sum(sentiment score for each term in tweet) for each tweet
    # append tweetSentiment score to each newTerm
    for tweet in tweet_file:
        tweetIx += 1
        # if (tweetIx < 10): continue
        # if (tweetIx > 20): break

        tweetJson = json.loads(tweet)
        tweetSentiment = 0
        if (("entities" in tweetJson.keys()) and
            (len(tweetJson['entities']['hashtags']) > 0) and
            # (tweetJson['place']['country_code'] == 'US') and
            True):
            # print "tweetIx: %d: tweetJson.keys: %s" % (tweetIx, tweetJson.keys())
            # print "tweetIx: %d: tweetJson.keys() has 'coordinates': %s" % \
            #       (tweetIx, str('coordinates' in tweetJson.keys()))
            # print "tweetIx: %d: tweetJson['entities']:" % (tweetIx)
            # pprint.pprint(tweetJson['entities'])
            # print "tweetIx: %d: tweetJson['entities']['hashtags']:" % (tweetIx)
            # pprint.pprint(tweetJson['entities']['hashtags'])

            tweetHashtags = [hashtagDict['text'] for hashtagDict in tweetJson['entities']['hashtags']]
            # print "tweetIx: %d: hashtags: " % (tweetIx)
            # pprint.pprint(tweetHashtags)
            for hashtag in tweetHashtags:
                hashTagFreqs = putHashtag(hashtag, hashTagFreqs)
            # tweetSentiment += sum(map(getSentiment, tweetTokens))
            # stateSentiments = putSentiment(tweetStateCd, tweetSentiment, stateSentiments)

        #print str(tweetSentiment)

    #pprint.pprint(hashTagFreqs)
    hashtagTuples = [(hashTagFreqs[hashtag], hashtag) \
                        for hashtag in hashTagFreqs.keys()]
    hashtagTuples = sorted(hashtagTuples, reverse = True)
    #pprint.pprint(hashtagTuples[0:19])
    #print "%d" % (min(10, len(hashtagTuples) - 1))
    for hashtagTuple in hashtagTuples[0:min(10, len(hashtagTuples) - 1)]:
        print "%s %d" % (hashtagTuple[1], hashtagTuple[0])

if __name__ == '__main__':
    main()

#os.chdir("/Users/bbalaji-2012/Documents/Work/Courses/Coursera/uwashington-datascience/assignment1");
#sys.argv[1] = "AFINN-111.txt"; sys.argv[2] = "output.txt"
#print(sys.argv[1]); print(sys.argv[2])
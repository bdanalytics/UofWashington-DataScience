import sys
import os
import pprint
import json
import re
#import numpy as np


def main():
    tweet_file = open(sys.argv[1])

    termFreqs = {}
    
    def updFrequency(term, termFreqs):
        try:
            junk = term.encode("UTF-8")
        except UnicodeDecodeError:
            term = term.decode("UTF-8")

        # print "putSentiment: term: %s" % (term)
        if (term in termFreqs.keys()): # update existing count
            # print "putSentiment: term: %s: (before appending)" % (term)
            # pprint.pprint(termFreqs[term])
            # print "putSentiment: term: %s: appending: %d" % (term, tweetSentiment)
            termFreqs[term] += 1
            # print "putSentiment: term: %s: (after  appending)" % (term)
            # pprint.pprint(termFreqs[term])
            return(termFreqs)
        else: # add key
            termFreqs[term] = 1
            return(termFreqs)

    tweetIx = 0  # used for debugging only

    # compute sum(sentiment score for each term in tweet) for each tweet
    # append tweetSentiment score to each newTerm
    for tweet in tweet_file:
        tweetIx += 1
        # if (tweetIx < 10): continue
        # if (tweetIx > 20): break

        tweetJson = json.loads(tweet)
        if ("text" in tweetJson.keys()):
            tweetTokens = [token for token in re.split('\W+', tweetJson['text']) if token != u'']
            for term in tweetTokens:
                termFreqs = updFrequency(term, termFreqs)

    # output frequency for each term
    for term in termFreqs.keys():
        # print "%s %f" % (newTerm, np.mean(newTerms[newTerm]))
        print "%s %f" % (term, termFreqs[term])


if __name__ == '__main__':
    main()

#os.chdir("/Users/bbalaji-2012/Documents/Work/Courses/Coursera/uwashington-datascience/assignment1")
#sys.argv[1] = "problem_1_submission.txt"
#print(sys.argv[1]); print(sys.argv[2])
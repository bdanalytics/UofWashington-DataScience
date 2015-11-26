import sys
import os
import pprint
import json
import re
#import numpy as np


def main():
    afinnfile = open(sys.argv[1])
    tweet_file = open(sys.argv[2])

    scores = {} # initialize an empty dictionary
    for line in afinnfile:
        term, score  = line.split("\t")  # The file is tab-delimited. "\t" means "tab character"
        try:
            junk = term.encode("UTF-8")
        except UnicodeDecodeError:
            term = term.decode("UTF-8")

        scores[term] = int(score)  # Convert the score to an integer.

    #print scores.items() # Print every (term, score) pair in the dictionary

    # get pre-specified sentiment score
    def getSentiment(term):
        try:
            junk = term.encode("UTF-8")
        except UnicodeDecodeError:
            term = term.decode("UTF-8")

        #print "getSentiment: term: %s; term.encode: %s" % (term, term.encode("UTF-8"))
        if (term in scores.keys()):
            return(scores[term])
        else:
            return(0)

    newTerms = {} # initialize

    # get pre-specified sentiment score
    def putSentiment(term, newTerms):
        try:
            junk = term.encode("UTF-8")
        except UnicodeDecodeError:
            term = term.decode("UTF-8")

        # print "putSentiment: term: %s" % (term)
        if (term in newTerms.keys()): # append to existing list
            # print "putSentiment: term: %s: (before appending)" % (term)
            # pprint.pprint(newTerms[term])
            # print "putSentiment: term: %s: appending: %d" % (term, tweetSentiment)
            newTerms[term].append(tweetSentiment)
            # print "putSentiment: term: %s: (after  appending)" % (term)
            # pprint.pprint(newTerms[term])
            return(newTerms)
        else: # add key
            newTerms[term] = [tweetSentiment]
            return(newTerms)

    tweetIx = 0  # used for debugging only

    # compute sum(sentiment score for each term in tweet) for each tweet
    # append tweetSentiment score to each newTerm
    for tweet in tweet_file:
        tweetIx += 1
        # if (tweetIx < 10): continue
        # if (tweetIx > 20): break

        tweetJson = json.loads(tweet)
        tweetSentiment = 0
        if ("text" in tweetJson.keys()):
            # print "tweetIx: %d: text: %s" % (tweetIx, tweetJson['text'])
            tweetTokens = [token for token in re.split('\W+', tweetJson['text']) if token != u'']
            # print "tweetIx: %d: tokens: " % (tweetIx)
            # pprint.pprint(tweetTokens)
            tweetSentiment += sum(map(getSentiment, tweetTokens))
            tweetNewTerms = list(set(tweetTokens) - set(scores.keys()))
            #addedTerms = sum(map(putSentiment, tweetNewTerms))
            for newTerm in tweetNewTerms:
                newTerms = putSentiment(newTerm, newTerms)

        #print str(tweetSentiment)

    # output mean(tweetSentiment for each tweet) for each newTerm
    for newTerm in newTerms.keys():
        # print "%s %f" % (newTerm, np.mean(newTerms[newTerm]))
        print "%s %f" % (newTerm, sum(newTerms[newTerm]) * 1.0 / len(newTerms[newTerm]))


if __name__ == '__main__':
    main()

#os.chdir("/Users/bbalaji-2012/Documents/Work/Courses/Coursera/uwashington-datascience/assignment1"); sys.argv[1] = "AFINN-111.txt"; sys.argv[2] = "output.txt"
#print(sys.argv[1]); print(sys.argv[2])
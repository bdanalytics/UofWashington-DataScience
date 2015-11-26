import sys
import os
import pprint
import json
import re

def hw():
    print 'Hello, world!'

def lines(fp):
    print str(len(fp.readlines()))

def main():
    #sent_file = open(sys.argv[1])
    tweet_file = open(sys.argv[2])
    #hw()
    #lines(sent_file)
    #lines(tweet_file)

    #afinnfile = open("AFINN-111.txt")
    afinnfile = open(sys.argv[1])
    scores = {} # initialize an empty dictionary
    for line in afinnfile:
        term, score  = line.split("\t")  # The file is tab-delimited. "\t" means "tab character"
        scores[term] = int(score)  # Convert the score to an integer.

    #print scores.items() # Print every (term, score) pair in the dictionary

    def getSentiment(word):
        #print "getSentiment: word: %s" % (word)
        if (word in scores.keys()):
            return(scores[word])
        else:
            return(0)

    lineIx = 0
    for line in tweet_file:
        lineIx += 1
        # if (lineIx < 10): continue
        # if (lineIx > 20): break
        tweetJson = json.loads(line)
        #print "tweet ix: %d; tweetJson: %s" % (lineIx, tweetJson.items())
        #print "tweet ix: %d; tweetJson.keys: %s" % (lineIx, tweetJson.keys())
        tweetSentiment = 0
        if ("text" in tweetJson.keys()):
            # print "tweet ix: %d; tweetJson['text']: %s" % (lineIx, tweetJson['text'])
            tweetTokens = re.split('\W+', tweetJson['text'])
            # print "tweet ix: %d; tweetTokens: %s" % (lineIx, str(tweetTokens))
            # pprint.pprint(tweetTokens)
            # print "tweet ix: %d; tweetTokens.sentiment: %s" % \
            #       (lineIx, map(getSentiment, tweetTokens))
            tweetSentiment += sum(map(getSentiment, tweetTokens))

        print str(tweetSentiment)
        #print "tweet ix: %d; tweetJson['text']: %s" % (lineIx, tweetJson['text'])

if __name__ == '__main__':
    main()

#os.chdir(os.getcwd() + "/assignment1"); sys.argv[1] = "AFINN-111.txt"; sys.argv[2] = "output.txt"
#print(sys.argv[1]); print(sys.argv[2])
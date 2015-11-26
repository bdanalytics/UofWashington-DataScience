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

    stateCds = {
            'AK': 'Alaska',
            'AL': 'Alabama',
            'AR': 'Arkansas',
            'AS': 'American Samoa',
            'AZ': 'Arizona',
            'CA': 'California',
            'CO': 'Colorado',
            'CT': 'Connecticut',
            'DC': 'District of Columbia',
            'DE': 'Delaware',
            'FL': 'Florida',
            'GA': 'Georgia',
            'GU': 'Guam',
            'HI': 'Hawaii',
            'IA': 'Iowa',
            'ID': 'Idaho',
            'IL': 'Illinois',
            'IN': 'Indiana',
            'KS': 'Kansas',
            'KY': 'Kentucky',
            'LA': 'Louisiana',
            'MA': 'Massachusetts',
            'MD': 'Maryland',
            'ME': 'Maine',
            'MI': 'Michigan',
            'MN': 'Minnesota',
            'MO': 'Missouri',
            'MP': 'Northern Mariana Islands',
            'MS': 'Mississippi',
            'MT': 'Montana',
            'NA': 'National',
            'NC': 'North Carolina',
            'ND': 'North Dakota',
            'NE': 'Nebraska',
            'NH': 'New Hampshire',
            'NJ': 'New Jersey',
            'NM': 'New Mexico',
            'NV': 'Nevada',
            'NY': 'New York',
            'OH': 'Ohio',
            'OK': 'Oklahoma',
            'OR': 'Oregon',
            'PA': 'Pennsylvania',
            'PR': 'Puerto Rico',
            'RI': 'Rhode Island',
            'SC': 'South Carolina',
            'SD': 'South Dakota',
            'TN': 'Tennessee',
            'TX': 'Texas',
            'UT': 'Utah',
            'VA': 'Virginia',
            'VI': 'Virgin Islands',
            'VT': 'Vermont',
            'WA': 'Washington',
            'WI': 'Wisconsin',
            'WV': 'West Virginia',
            'WY': 'Wyoming'
    }
    stateNames = {} # Reverse dictionary
    for code in stateCds.keys():
        stateNames[stateCds[code]] = code

    stateSentiments = {}

    # get pre-specified sentiment score
    def putSentiment(stateCd, tweetSentiment, stateSentiments):
        stateCd = stateCd.encode("UTF-8")
        if (stateCd in stateSentiments.keys()): # append to existing list
            stateSentiments[stateCd].append(tweetSentiment)
            return(stateSentiments)
        else: # add key
            stateSentiments[stateCd] = [tweetSentiment]
            return(stateSentiments)

    tweetIx = 0  # used for debugging only

    # compute sum(sentiment score for each term in tweet) for each tweet
    # append tweetSentiment score to each newTerm
    for tweet in tweet_file:
        tweetIx += 1
        # if (tweetIx < 10): continue
        # if (tweetIx > 20): break

        tweetJson = json.loads(tweet)
        tweetSentiment = 0
        if (("text" in tweetJson.keys()) and
            (tweetJson['place'] != None) and
            (tweetJson['place']['country_code'] == 'US')):
            # print "tweetIx: %d: tweetJson.keys: %s" % (tweetIx, tweetJson.keys())
            # print "tweetIx: %d: tweetJson.keys() has 'coordinates': %s" % \
            #       (tweetIx, str('coordinates' in tweetJson.keys()))
            # print "tweetIx: %d: tweetJson['coordinates']:" % (tweetIx)
            # pprint.pprint(tweetJson['coordinates'])
            # print "tweetIx: %d: tweetJson['place']:" % (tweetIx)
            # pprint.pprint(tweetJson['place'])
            tweetPlaceFullName = tweetJson['place']['full_name']
            tweetPlaceFullNameTokens = re.split('\W+', tweetJson['place']['full_name'])
            tweetStateCd = tweetPlaceFullNameTokens[-1]
            if (tweetStateCd == "USA"): # tokens before contain fully spelled state names
                tweetStateName = ' '.join(tweetPlaceFullNameTokens[:-1])
                tweetStateCd = stateNames[tweetStateName]

            # print "tweetIx: %d; tweetPlaceFullName: %s; tweetStateCd: %s" % \
            #       (tweetIx, tweetPlaceFullName, tweetStateCd)
            # print "tweetIx: %d: text: %s" % (tweetIx, tweetJson['text'])
            tweetTokens = [token for token in re.split('\W+', tweetJson['text']) if token != u'']
            # print "tweetIx: %d: tokens: " % (tweetIx)
            # pprint.pprint(tweetTokens)
            tweetSentiment += sum(map(getSentiment, tweetTokens))
            stateSentiments = putSentiment(tweetStateCd, tweetSentiment, stateSentiments)

        #print str(tweetSentiment)

    #pprint.pprint(stateSentiments)
    stateTuples = [(sum(stateSentiments[stateCd]) * 1.0 / len(stateSentiments[stateCd]), stateCd) \
                   for stateCd in stateSentiments.keys()]
    stateTuples = sorted(stateTuples, reverse = True)
    #pprint.pprint(stateTuples)
    print stateTuples[0][1]

if __name__ == '__main__':
    main()

#os.chdir("/Users/bbalaji-2012/Documents/Work/Courses/Coursera/uwashington-datascience/assignment1");
#sys.argv[1] = "AFINN-111.txt"; sys.argv[2] = "output.txt"
#print(sys.argv[1]); print(sys.argv[2])
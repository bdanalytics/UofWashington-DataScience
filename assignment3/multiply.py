import sys
# import os
#os.chdir("/Users/bbalaji-2012/Documents/Work/Courses/Coursera/uwashington/uwashington-datascience/assignment3");

import MapReduce
# import imp
# MapReduce = imp.load_source('MapReduce', 'MapReduce.py')

"""
Problem 6: Sparse matrrices multiplication Example in the Simple Python MapReduce Framework
sys.argv[1] = "data/matrix.json"
"""

mr = MapReduce.MapReduce()

# =============================
# Do not modify above this line

def mapper(record):
    # record: (matrixId, i, j, value)

    # key: i, j of AB:
    #   if matrixId is A: i, k = 0:5 (hard-coded)
    #   if matrixId is B: k = 0:5 (hard-coded), j
    # value: (matrixId, value)

    if record[0] == "a":
        for k in range(0, 5):
            mr.emit_intermediate((record[1], k), ("a", record[2], record[3]))
    elif record[0] == "b":
        for k in range(0, 5):
            mr.emit_intermediate((k, record[2]), ("b", record[1], record[3]))
    else:
        mr.emit_intermediate("error: this should not happen")

def reducer(key, valueLst):
    # key: i, j of AB
    # value in valueLst: ("a" / "b", value)

    # mr.emit((key, valueLst))
    AValueLst = [val for val in valueLst if val[0] == "a"]
    # mr.emit((key, ("a", AValueLst)))
    AValueVct = [0] * 5
    for e in AValueLst:
        AValueVct[e[1]] = e[2]

    # mr.emit((key, ("a", AValueVct)))

    BValueLst = [val for val in valueLst if val[0] == "b"]
    # mr.emit((key, ("b", BValueLst)))
    BValueVct = [0] * 5
    for e in BValueLst:
        BValueVct[e[1]] = e[2]

    # mr.emit((key, ("b", BValueVct)))

    # len(AValueLst) == len(BValueLst)
    ABVal = sum([AValueVct[ix] * BValueVct[ix] for ix in range(0, len(AValueVct))])
    mr.emit((key[0], key[1], ABVal))

# import json
# lineNum = 1
# for line in inputdata:
#     print line
#     record = json.loads(line)
#     print record
#     lineNum += 1
#     if lineNum > 5:
#         break

# Do not modify below this line
# =============================
if __name__ == '__main__':
  inputdata = open(sys.argv[1])
  mr.execute(inputdata, mapper, reducer)

#os.chdir("/Users/bbalaji-2012/Documents/Work/Courses/Coursera/uwashington/uwashington-datascience/assignment3");
#sys.argv[1] = "books.json"; sys.argv[2] = "output.txt"
#print(sys.argv[1]); print(sys.argv[2])
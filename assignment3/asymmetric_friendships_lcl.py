import sys
import os
#os.chdir("/Users/bbalaji-2012/Documents/Work/Courses/Coursera/uwashington/uwashington-datascience/assignment3");

#import MapReduce
import imp
MapReduce = imp.load_source('MapReduce', 'MapReduce.py')

"""
Problem 2: Relational Join Example in the Simple Python MapReduce Framework
sys.argv[1] = "data/friends.json"
"""

mr = MapReduce.MapReduce()

# =============================
# Do not modify above this line

def mapper(record):
    # key: personA
    # value: ("B", personB)

    key = record[0]
    value = ("B", record[1])
    mr.emit_intermediate(key, value)

    # key: personB
    # value: ("A", personA)

    key = record[1]
    value = ("A", record[0])
    mr.emit_intermediate(key, value)

def reducer(key, valueLst):

    # personALst = [tpl[1] for tpl in valueLst if tpl[0] == "A"]
    # personBLst = [tpl[1] for tpl in valueLst if tpl[0] == "B"]

    personASet = set([tpl[1] for tpl in valueLst if tpl[0] == "A"])
    personBSet = set([tpl[1] for tpl in valueLst if tpl[0] == "B"])
    # mr.emit((key, ("A", list(personASet)), ("B", list(personBSet))))

    # ABDiff = personASet.difference(personBSet)
    # mr.emit((key, ("ABDiff", list(ABDiff))))
    # for personA in personASet:
    #     mr.emit((personA, key))

    # BADiff = personBSet.difference(personASet)
    # mr.emit((key, ("BABDiff", list(BADiff))))
    # for personB in personBSet:
    #     mr.emit((key, personB))

    SymDiff = personASet.symmetric_difference(personBSet)
    # mr.emit((key, ("SymDiff", list(SymDiff))))
    for person in SymDiff:
        mr.emit((key, person))

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
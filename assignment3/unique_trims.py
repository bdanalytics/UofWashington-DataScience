import sys
# import os
# os.chdir("/Users/bbalaji-2012/Documents/Work/Courses/Coursera/uwashington/uwashington-datascience/assignment3");

import MapReduce
# import imp
# MapReduce = imp.load_source('MapReduce', 'MapReduce.py')

"""
Problem 5: Nucleotide trim Example in the Simple Python MapReduce Framework
sys.argv[1] = "data/dna.json"
"""

mr = MapReduce.MapReduce()

# =============================
# Do not modify above this line

def mapper(record):
    # key: trimmed nucleotide
    # value: count of trimmed nucleotide to dedup in reduce

    key = record[1][:-10]
    # value = 1
    mr.emit_intermediate(key, 1)

def reducer(key, valueLst):
    # key: trimmed nucleotide
    # value in valueLst: 1 for each occurence

    mr.emit((key))

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
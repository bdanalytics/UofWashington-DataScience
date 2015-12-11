import sys
"""
import os
#os.chdir("/Users/bbalaji-2012/Documents/Work/Courses/Coursera/uwashington/uwashington-datascience/assignment3");
"""

import MapReduce
"""
import imp
MapReduce = imp.load_source('MapReduce', 'MapReduce.py')
"""

"""
Problem 1: Inverted Index Example in the Simple Python MapReduce Framework
sys.argv[1] = "data/books.json"
"""

mr = MapReduce.MapReduce()

# =============================
# Do not modify above this line

def mapper(record):
    # key: document identifier
    # value: document contents

    key = record[0]
    value = record[1]
    words = value.split()
    for w in words:
      mr.emit_intermediate(w, key)

def reducer(key, list_of_values):
    # key: word
    # value: list of document occurrences

    # for v in list_of_values:
    #   total += v
    mr.emit((key, list(set(list_of_values))))

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
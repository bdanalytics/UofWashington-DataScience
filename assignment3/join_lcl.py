import sys
import os
#os.chdir("/Users/bbalaji-2012/Documents/Work/Courses/Coursera/uwashington/uwashington-datascience/assignment3");

#import MapReduce
import imp
MapReduce = imp.load_source('MapReduce', 'MapReduce.py')

"""
Problem 2: Relational Join Example in the Simple Python MapReduce Framework
sys.argv[1] = "data/records.json"
"""

mr = MapReduce.MapReduce()

# =============================
# Do not modify above this line

def mapper(record):
    # key: order_id
    # value: table record contents

    key = record[1]
    value = record
    mr.emit_intermediate(key, record)

def reducer(key, valueLst):

#     result = []
#
#     if (len(valueLst) == 2):
#         for value in valueLst:
#             result.extend(value)
#
#     else:
#         result.extend([(key, len(valueLst), [(value[0], value[2]) for value in valueLst])])
#
#     mr.emit(result)

    # for orderVal in valueLst:
    #     if (orderVal[0] == "order"):
    #         orderInfo = orderVal
    #         for lineVal in valueLst:
    #             if (lineVal[0] == "line_item"):
    #                 # result = orderInfo
    #                 result.extend(lineVal)
    #                 mr.emit(result)
    #             else:
    #                 result = []
    #                 result.extend([(key, len(valueLst), [(value[0], value[2]) for value in valueLst])])
    #                 mr.emit(result)

    # for orderVal in valueLst:
    #     if (orderVal[0] == "order"):
    #         orderInfo = orderVal
    #         for lineVal in valueLst:
    #             if (lineVal[0] == "line_item"):
    #                 # result = orderInfo
    #                 result.extend(lineVal)
    #                 mr.emit(result)
    #             else:
    #                 result = []
    #                 result.extend([(key, len(valueLst), [(value[0], value[2]) for value in valueLst])])
    #                 mr.emit(result)

    orders = [value for value in valueLst if value[0] == "order"]
    #mr.emit((key, orders))
    line_items = [value for value in valueLst if value[0] == "line_item"]
    #mr.emit((key, line_items))

    for order in orders:
        for line_item in line_items:
            # Simple assignment just creates a pointer to result
            # result = order
            result = [elem for elem in order]
            result.extend(line_item)
            # result.extend(line_item[0:3])
            mr.emit(result)

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
import os
import pickle
from collections import defaultdict
from itertools import groupby
from operator import itemgetter
from nltk.parse import CoreNLPParser
from nltk import pos_tag
from gensim.similarities import WmdSimilarity
import gensim
import datetime

def read_relation(path):
    """ Read relation files and process

        (str) -> (dict)
    """

    file_path = os.path.join(os.pardir, "out", path, "relations.pkl")
    relations_file = open(file_path, 'rb')
    relations = pickle.load(relations_file)
    relations_file.close()
    return relations


# Read comparative sentences
stackoverflow = read_relation("stackoverflow_v1")

flag = True
edges = stackoverflow.keys()

tag_groups = []

# for pair in tag_pairs:
#     flag = False
#     for p in pair:
#         if tag_groups:
#             for s in tag_groups:
#                 if p in s:
#                     for tag in pair:
#                         s.add(tag)
#                     flag = True
#                     break
#     if not flag:
#         tag_groups.append(set([pair[0], pair[1]]))
#         flag = False


def dfs(adj_list, visited, vertex, result, key):
    visited.add(vertex)
    result[key].append(vertex)
    for neighbor in adj_list[vertex]:
        if neighbor not in visited:
            dfs(adj_list, visited, neighbor, result, key)
#
# edges = [('c', 'e'), ('c', 'd'), ('a', 'b'), ('d', 'e'), ('b','a') ]

adj_list = defaultdict(list)
for x, y in edges:
    adj_list[x].append(y)
    adj_list[y].append(x)



result = defaultdict(list)
visited = set()
for vertex in adj_list:
    if vertex not in visited:
        dfs(adj_list, visited, vertex, result, vertex)


for row in result.values():
    print(row)

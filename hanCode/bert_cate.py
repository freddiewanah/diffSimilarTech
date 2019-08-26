
from bert_serving.client import BertClient

import os
import pickle
from nltk.parse import CoreNLPParser
from nltk import pos_tag
from gensim.similarities import WmdSimilarity
import gensim
import datetime
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import json
import psycopg2
import operator

# cate_bert_num_v2 = {
#     "algorithm": 6,
#     "application": 7,
#     "browser": 10,
#     "class": 10,
#     "dbms": 6,
#     "editor": 5,
#     "engine": 6,
#     "format": 4,
#     "framework": 7,
#     "function": 5,
#     "ide": 3,
#     "language": 6,
#     "library": 6,
#     "method": 8,
#     "operation": 11,
#     "os": 4,
#     "package": 4,
#     "protocol": 6,
#     "software": 5,
#     "structure": 7,
#     "system": 6
# }

# cate_bert_num_v3 = {
#     "algorithm": 5,
#     "application": 5,
#     "browser": 9,
#     "class": 9,
#     "dbms": 6,
#     "editor": 6,
#     "engine": 5,
#     "format": 7,
#     "framework": 5,
#     "function": 5,
#     "ide": 3,
#     "language": 8,
#     "library": 6,
#     "method": 7,
#     "operation": 9,
#     "os": 4,
#     "package": 12,
#     "protocol": 5,
#     "software": 8,
#     "structure": 7,
#     "system": 4
# }

cate_bert_num_v4 = {
    "algorithm": 5,
    "application": 4,
    "browser": 7,
    "class": 9,
    "dbms": 5,
    "editor": 6,
    "engine": 5,
    "format": 5,
    "framework": 8,
    "function": 6,
    "ide": 3,
    "language": 8,
    "library": 5,
    "method": 9,
    "operation": 15,
    "os": 4,
    "package": 11,
    "protocol": 4,
    "software": 6,
    "structure": 9,
    "system": 5
}



cate_list = ["library", "class", "function", "framework", "language", "system", "os", "method", "operation",
             "protocol", "editor", "format", "algorithm", "structure", "database", "dbms", "app", "application",
             "package", "ide", "browser", "engine", "software"]
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
stack = read_relation("stackoverflow_v1")
relations = [stack]
def add_dict(dictionary, word):
    """ Record word.

        (dict, str) -> None
    """
    if word in dictionary:
        dictionary[word] += 1
    else:
        dictionary[word] = 1


# Init category
categories = {}
techs = []
tags = {}
other = {}
word_dict = {}

cate_count = {}

print(datetime.datetime.now())
for relation in relations:

    for pair in relation:
        sentences = list(relation[pair])
        cp = [sentence[-1] for sentence in sentences]
        cp_new = []
        for sentence in cp:
            sentence = sentence.replace(pair[0], "*")
            sentence = sentence.replace(pair[1], "*")
            if sentence not in cp_new:
                cp_new.append(sentence)
        conn = psycopg2.connect('dbname=stackoverflow port=5433 host=localhost')
        cursor = conn.cursor()
        query = "SELECT category FROM {} WHERE tag = '{}' OR tag = '{}'".format(
            "tag_cate", pair[0], pair[1])
        cursor.execute(query)
        row = cursor.fetchall()

        if row != []:
            techs.append(pair[0])
            techs.append(pair[1])

            for cate in row:
                add_dict(cate_count, cate)
                if cate[0] in cate_list:

                    if cate[0] in categories.keys():
                        categories[cate[0]] += cp_new
                    else:
                        categories[cate[0]] = [i for i in cp_new]

cate_clean = {}
for a in categories.keys():
    cate_clean[a] = list(dict.fromkeys(categories[a]))
with open(os.path.join(os.pardir, "outnew", "cate", "bert_category.json"), 'w') as fp:
    json.dump(cate_clean, fp)

file = open(os.path.join(os.pardir, "outnew", "cate", "bert_category.json"), 'r')
cates = json.load(file)

# sorted_cate = sorted(cate_count.items(), key=operator.itemgetter(1), reverse=True)

count = 0
# for c in cate_list:
#     print("Start cate: {}. {} Sentences".format(c, len(cates[c])))
#     count+=1
#     corpus = cates[c]
#     bc = BertClient()
#     corpus_cleared = []
#     for sen in corpus:
#         corpus_cleared.append(sen.replace("*", ""))
#     vec = bc.encode(corpus_cleared)
#     path = os.path.join(os.pardir, "outnew", "bert_cate_v4", "{}_vec".format(c))
#     np.save(path, vec)
#     plt.figure(figsize=(10, 7))
#
#     dendrogram = sch.dendrogram(sch.linkage(vec, method='ward'))
#     png_path = os.path.join(os.pardir, "outnew", "bert_cate_v4", "{}_HC.png".format(c))
#     plt.savefig(png_path)
#     print(datetime.datetime.now())
#     print("Finished cate: {} --- {}/{}".format(c, count, len(cate_list)))

for key in cate_bert_num_v4.keys():
    print(datetime.datetime.now())
    print("Start cate: {}".format(key))
    vec = np.load(os.path.join(os.pardir, "outnew", "bert_cate_v4", "{}_vec.npy".format(key)))

    dendrogram = sch.dendrogram(sch.linkage(vec, method='ward'))
    cluster = AgglomerativeClustering(n_clusters=cate_bert_num_v4 [key], affinity='euclidean', linkage='ward')
    clusters = cluster.fit_predict(vec)
    for index, val in enumerate(clusters):
        newpath = os.path.join(os.pardir, "outnew", "bert_cate_v4", "{}".format(key))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        out_file = open(os.path.join(os.pardir, "outnew", "bert_cate_v4", "{}".format(key), "{}.txt".format(val)), "a")
        out_file.write(cates[key][index])
        out_file.write("\n")
        out_file.close()

print(datetime.datetime.now())

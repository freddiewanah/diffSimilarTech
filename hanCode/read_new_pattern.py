import os
import pickle
import psycopg2
import json
from nltk.parse import CoreNLPParser
from nltk import pos_tag
import operator
import datetime

def readFile(fileName):
    idFlag = False
    tempList = []
    tempSet = []
    with open(os.path.join(os.pardir, "outFinal", fileName)) as leased_pattern_file:
        for line in leased_pattern_file:
            line = line.rstrip()
            if line == "":
                idFlag = False
                if len(tempSet) != 0:
                    tempList.append(tempSet)
                    tempSet = []
            if idFlag:
                tempSet.append(line)
            if RepresentsInt(line):
                idFlag = True
                tempSet.append(line)
    return tempList

def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


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


c = 0
for pair in stackoverflow.keys():
    for i in stackoverflow[pair]:
        if int(i[-2]) > c:
            c = int(i[-2])


print(c)
#
# # read leased pattern 0,1
# tList = readFile("leased_pattern.txt")
# print(len(tList))
#
# for temp in tList:
#     keyPair = temp[1].split(' ')
#     keyA = (keyPair[0], keyPair[1])
#     keyB = (keyPair[1], keyPair[0])
#     sen = ';'.join(t for t in temp[3:])
#     # print(sen)
#     if keyA in stackoverflow.keys():
#         stackoverflow[keyA].add((keyPair[0], '', keyPair[1], '', temp[0], sen))
#
#     if keyB in stackoverflow.keys():
#         stackoverflow[keyB].add((keyPair[0], '', keyPair[1], '', temp[0], sen))
#
#     if keyA not in stackoverflow.keys() and keyB not in stackoverflow.keys():
#         stackoverflow[keyA] = set()
#         stackoverflow[keyA].add((keyPair[0], '', keyPair[1], '', temp[0], sen))
#
# # read not done
#
#
#
# tList = readFile("leased_not_do.txt")
# print(len(tList))
#
# for temp in tList:
#     keyPair = temp[1].split(' ')
#     keyA = (keyPair[0], keyPair[1])
#     keyB = (keyPair[1], keyPair[0])
#     sen = ';'.join(t for t in temp[3:])
#     # print(sen)
#     if keyA in stackoverflow.keys():
#         stackoverflow[keyA].add((keyPair[0], '', keyPair[1], '', temp[0], sen))
#
#     if keyB in stackoverflow.keys():
#         stackoverflow[keyB].add((keyPair[0], '', keyPair[1], '', temp[0], sen))
#
#     if keyA not in stackoverflow.keys() and keyB not in stackoverflow.keys():
#         stackoverflow[keyA] = set()
#         stackoverflow[keyA].add((keyPair[0], '', keyPair[1], '', temp[0], sen))
#
#
# # read leased old
#
# tList = readFile("leased_old.txt")
# print(len(tList))
#
# for temp in tList:
#     keyPair = temp[1].split('\t')
#     keyA = (keyPair[0], keyPair[1])
#     keyB = (keyPair[1], keyPair[0])
#     sen = temp[3]
#     # print(sen)
#     if keyA in stackoverflow.keys():
#         stackoverflow[keyA].add((keyPair[0], '', keyPair[1], '', temp[0], sen))
#
#     if keyB in stackoverflow.keys():
#         stackoverflow[keyB].add((keyPair[0], '', keyPair[1], '', temp[0], sen))
#
#     if keyA not in stackoverflow.keys() and keyB not in stackoverflow.keys():
#         stackoverflow[keyA] = set()
#         stackoverflow[keyA].add((keyPair[0], '', keyPair[1], '', temp[0], sen))
#
# output = open(os.path.join(os.pardir, 'outFinal', 'all_sentences.pkl'), 'wb')
# pickle.dump(stackoverflow, output)
# output.close()
#

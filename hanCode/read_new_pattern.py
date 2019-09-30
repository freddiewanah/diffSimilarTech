import os
import pickle
import psycopg2
import json
from nltk.parse import CoreNLPParser
from nltk import pos_tag
import operator
import datetime

def read_relation_old(path):
    """ Read relation files and process

        (str) -> (dict)
    """

    file_path = os.path.join(os.pardir, "out", path, "relations.pkl")
    relations_file = open(file_path, 'rb')
    relations = pickle.load(relations_file)
    relations_file.close()
    return relations


# Read comparative sentences
stackoverflow_old = read_relation_old("stackoverflow_v1")
c = 0
for i in stackoverflow_old.keys():
    c += len(stackoverflow_old[i])

print(c)

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


def read_relation():
    """ Read relation files and process

        (str) -> (dict)
    """

    file_path = os.path.join(os.pardir, "outFinal", "all_sentences_old.pkl")
    relations_file = open(file_path, 'rb')
    relations = pickle.load(relations_file)
    relations_file.close()
    return relations


# Read comparative sentences
stackoverflow = read_relation()

print(len(stackoverflow.keys()))
# c = 0
# for pair in stackoverflow.keys():
#     for i in stackoverflow[pair]:
#         if int(i[-2]) > c:
#             c = int(i[-2])
#
#
# print(c)

stackoverflow = {}

# read leased pattern 0,1
tList = readFile("pattern_17422.txt")
print(len(tList))

for temp in tList:
    keyPair = temp[1].split(' ')
    keyA = (keyPair[0], keyPair[1])
    keyB = (keyPair[1], keyPair[0])
    sen = ';'.join(t for t in temp[3:])
    # print(sen)
    if keyA in stackoverflow.keys():
        stackoverflow[keyA].add((keyPair[0], '', keyPair[1], '', temp[0], sen))

    if keyB in stackoverflow.keys():
        stackoverflow[keyB].add((keyPair[0], '', keyPair[1], '', temp[0], sen))

    if keyA not in stackoverflow.keys() and keyB not in stackoverflow.keys():
        stackoverflow[keyA] = set()
        stackoverflow[keyA].add((keyPair[0], '', keyPair[1], '', temp[0], sen))

tList = readFile("leased_pattern.txt")
print(len(tList))

for temp in tList:
    keyPair = temp[1].split(' ')
    keyA = (keyPair[0], keyPair[1])
    keyB = (keyPair[1], keyPair[0])
    sen = ';'.join(t for t in temp[3:])
    # print(sen)
    if keyA in stackoverflow.keys():
        stackoverflow[keyA].add((keyPair[0], '', keyPair[1], '', temp[0], sen))

    if keyB in stackoverflow.keys():
        stackoverflow[keyB].add((keyPair[0], '', keyPair[1], '', temp[0], sen))

    if keyA not in stackoverflow.keys() and keyB not in stackoverflow.keys():
        stackoverflow[keyA] = set()
        stackoverflow[keyA].add((keyPair[0], '', keyPair[1], '', temp[0], sen))

# read not done
tList = readFile("pattern_s=0.txt")
print(len(tList))

for temp in tList:
    keyPair = temp[1].split(' ')
    keyA = (keyPair[0], keyPair[1])
    keyB = (keyPair[1], keyPair[0])
    sen = ';'.join(t for t in temp[3:])
    # print(sen)
    if keyA in stackoverflow.keys():
        stackoverflow[keyA].add((keyPair[0], '', keyPair[1], '', temp[0], sen))

    if keyB in stackoverflow.keys():
        stackoverflow[keyB].add((keyPair[0], '', keyPair[1], '', temp[0], sen))

    if keyA not in stackoverflow.keys() and keyB not in stackoverflow.keys():
        stackoverflow[keyA] = set()
        stackoverflow[keyA].add((keyPair[0], '', keyPair[1], '', temp[0], sen))


tList = readFile("not_do_leased_output_89581.txt")
print(len(tList))

for temp in tList:
    keyPair = temp[1].split(' ')
    keyA = (keyPair[0], keyPair[1])
    keyB = (keyPair[1], keyPair[0])
    sen = ';'.join(t for t in temp[3:])
    # print(sen)
    if keyA in stackoverflow.keys():
        stackoverflow[keyA].add((keyPair[0], '', keyPair[1], '', temp[0], sen))

    if keyB in stackoverflow.keys():
        stackoverflow[keyB].add((keyPair[0], '', keyPair[1], '', temp[0], sen))

    if keyA not in stackoverflow.keys() and keyB not in stackoverflow.keys():
        stackoverflow[keyA] = set()
        stackoverflow[keyA].add((keyPair[0], '', keyPair[1], '', temp[0], sen))

tList = readFile("leased_not_do.txt")
print(len(tList))

for temp in tList:
    keyPair = temp[1].split(' ')
    keyA = (keyPair[0], keyPair[1])
    keyB = (keyPair[1], keyPair[0])
    sen = ';'.join(t for t in temp[3:])
    # print(sen)
    if keyA in stackoverflow.keys():
        stackoverflow[keyA].add((keyPair[0], '', keyPair[1], '', temp[0], sen))

    if keyB in stackoverflow.keys():
        stackoverflow[keyB].add((keyPair[0], '', keyPair[1], '', temp[0], sen))

    if keyA not in stackoverflow.keys() and keyB not in stackoverflow.keys():
        stackoverflow[keyA] = set()
        stackoverflow[keyA].add((keyPair[0], '', keyPair[1], '', temp[0], sen))


# read leased old

tList = readFile("output_500000_leased_17422.txt")
print(len(tList))

for temp in tList:
    keyPair = temp[1].split('\t')
    keyA = (keyPair[0], keyPair[1])
    keyB = (keyPair[1], keyPair[0])
    sen = temp[3]
    # print(sen)
    if keyA in stackoverflow.keys():
        stackoverflow[keyA].add((keyPair[0], '', keyPair[1], '', temp[0], sen))

    if keyB in stackoverflow.keys():
        stackoverflow[keyB].add((keyPair[0], '', keyPair[1], '', temp[0], sen))

    if keyA not in stackoverflow.keys() and keyB not in stackoverflow.keys():
        stackoverflow[keyA] = set()
        stackoverflow[keyA].add((keyPair[0], '', keyPair[1], '', temp[0], sen))

tList = readFile("output_500000_leased_s=0.txt")
print(len(tList))

for temp in tList:
    keyPair = temp[1].split('\t')
    keyA = (keyPair[0], keyPair[1])
    keyB = (keyPair[1], keyPair[0])
    sen = temp[3]
    # print(sen)
    if keyA in stackoverflow.keys():
        stackoverflow[keyA].add((keyPair[0], '', keyPair[1], '', temp[0], sen))

    if keyB in stackoverflow.keys():
        stackoverflow[keyB].add((keyPair[0], '', keyPair[1], '', temp[0], sen))

    if keyA not in stackoverflow.keys() and keyB not in stackoverflow.keys():
        stackoverflow[keyA] = set()
        stackoverflow[keyA].add((keyPair[0], '', keyPair[1], '', temp[0], sen))

# read leased old

tList = readFile("output.txt")
print(len(tList))

for temp in tList:
    keyPair = temp[1].split('\t')
    keyA = (keyPair[0], keyPair[1])
    keyB = (keyPair[1], keyPair[0])
    sen = temp[3]
    pattern = temp[2]
    pList = ["Pattern(s): 5 5", "Pattern(s): 5","Pattern(s): 5  5	5"]
    if pattern in pList:
        continue
    if keyA in stackoverflow.keys():
        stackoverflow[keyA].add((keyPair[0], '', keyPair[1], '', temp[0], sen))

    if keyB in stackoverflow.keys():
        stackoverflow[keyB].add((keyPair[0], '', keyPair[1], '', temp[0], sen))

    if keyA not in stackoverflow.keys() and keyB not in stackoverflow.keys():
        stackoverflow[keyA] = set()
        stackoverflow[keyA].add((keyPair[0], '', keyPair[1], '', temp[0], sen))

tList = readFile("leased_old.txt")
print(len(tList))

for temp in tList:
    keyPair = temp[1].split('\t')
    keyA = (keyPair[0], keyPair[1])
    keyB = (keyPair[1], keyPair[0])
    sen = temp[3]
    pattern = temp[2]
    pList = ["Pattern(s): 5 5", "Pattern(s): 5","Pattern(s): 5  5	5"]
    if pattern in pList:
        continue
    if keyA in stackoverflow.keys():
        stackoverflow[keyA].add((keyPair[0], '', keyPair[1], '', temp[0], sen))

    if keyB in stackoverflow.keys():
        stackoverflow[keyB].add((keyPair[0], '', keyPair[1], '', temp[0], sen))

    if keyA not in stackoverflow.keys() and keyB not in stackoverflow.keys():
        stackoverflow[keyA] = set()
        stackoverflow[keyA].add((keyPair[0], '', keyPair[1], '', temp[0], sen))

print(len(stackoverflow))

print(len(stackoverflow.keys()))
newDict = {}
for i in stackoverflow.keys():
    sens = stackoverflow[i]
    newSens = set()

    for sen in sens:
        flag = False
        singleSen = sen[-1]
        idSen = sen[-2]
        for newSen in newSens:
            singleNewSen = newSen[-1]
            if newSen[-2] == idSen and singleSen == singleNewSen:
                flag = True
        if not flag:
            newSens.add(sen)
    newDict[i] = newSens
c = 0
for i in newDict.keys():
    c += len(newDict[i])
print(c)


notList = ["folder", "directory", "versioning", "upgrade", "cpu", "processors", "children", "parent"]

c = 0
for i in newDict.keys():
    if i[0] in notList and i[1] in notList:
        continue
    c += len(newDict[i])

print(c)




output = open(os.path.join(os.pardir, 'outFinal', 'new_sentences.pkl'), 'wb')
pickle.dump(stackoverflow, output)
output.close()


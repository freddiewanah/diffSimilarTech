import os
import pickle
import psycopg2
import json
from nltk.parse import CoreNLPParser
from nltk import pos_tag
import operator
import datetime
from random import shuffle

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
unix = read_relation("unix")
superuser = read_relation("superuser")
softwarerecs = read_relation("softwarerecs")

realtions = [stackoverflow, unix, superuser, softwarerecs]

# Init category
categories = {}
techs = []
tags = {}
other = {}

word_dict = {}

cate_count = {}

cate_list = ["library", "class", "function", "framework", "language", "system", "os", "method", "operation",
             "protocol", "editor", "format", "algorithm", "structure", "database", "dbms", "app", "application",
             "package", "ide", "browser", "engine", "software"]

not_word = ["is", "the", "than", "to", "s", "ve", "i", "a", "or", "and", "it", "of", "you", "be", "are", "more", "in",
            "with", "much", "but", "on", "has", "have", "for", "if", "n", "t", "so", "does", "that", "as", "which",
            "your"]


def get_pos_tag(words):
    """ Get POS tag of words.

        ([str]) -> ([str], [str])
    """
    tags = []
    flag = False
    tagged_words = CoreNLPParser(url='http://localhost:9000', tagtype='pos').tag(words)
    if len(words) != len(tagged_words):
        tagged_words = pos_tag(words)
    words = []
    for (word, tag) in tagged_words:
        if flag:
            word = "." + word
            flag = False

        tags.append(tag)
        words.append(word)
    return (words, tags)

def add_dict(dictionary, word):
    """ Record word.

        (dict, str) -> None
    """
    if word in dictionary:
        dictionary[word] += 1
    else:
        dictionary[word] = 1


def extract_topic(out_list, tag_list):
    """ Extract topic from the sentence.

        ([str]) -> str
    """
    # if len(out_list) == 1 and out_list[0] in ignore_set:
    #     return None
    # else:
    topic_list = ["NN", "NNP", "NNS", "NPS", "JJR", "JJ", "JJS", "RB", "RBR", "RBS"]
    if True:
        for i in range(len(out_list)):
            w = out_list[i]
            t = tag_list[i]
            if t in topic_list and w not in techs and w not in not_word:
                add_dict(tags, w)
            elif w not in techs and w not in not_word:
                add_dict(other, w)

            # if w in memory:
            #     return "memory"
            # elif w in usability:
            #     return "usability"
            # elif w in performance:
            #     return "performance"
            # elif w in security:
            #     return "security"
            # elif w in reliability:
            #     return "reliability"
        return ""

def main():
    print(datetime.datetime.now())
    s = []
    for relation in realtions:
        
        for pair in relation:
            sentences = list(relation[pair])
            cp = [sentence[-1] for sentence in sentences]

            # query = "SELECT category FROM {} WHERE tag = '{}' OR tag = '{}'".format(
            #     "tag_cate", pair[0], pair[1])
            # cursor.execute(query)
            # row = cursor.fetchall()
            #
            # if row != []:
            # print(pair[0], pair[1])
            for ss in cp:
                ss = ss.replace(pair[0], '11111111')
                ss = ss.replace(pair[1], '22222222')
                s.append(ss)

    shuffle(s)
    for t in s:
        print(t)
                # techs.append(pair[0])
                # techs.append(pair[1])


    #             for cate in row:
    #                 add_dict(cate_count, cate)
    #                 if cate[0] in cate_list:
    #
    #                     if cate[0] in categories.keys():
    #                         categories[cate[0]] += cp
    #                     else:
    #                         categories[cate[0]] = [i for i in cp]
    # with open(os.path.join(os.pardir, "outnew", "cate", "category.json"), 'w') as fp:
    #     json.dump(categories, fp)
    #
    # sorted_cate = sorted(cate_count.items(), key=operator.itemgetter(1), reverse=True)
    #
    # a = 0
    # b= 0
    # for c in cate_count:
    #     if c[0] in cate_list:
    #         a += cate_count[c]
    #     else:
    #         b += cate_count[c]
    #
    #
    # print(sorted_cate)
    # print(a,b)
    return
    for cate in categories:
        tags.clear()
        other.clear()
        for sent in categories[cate]:
            word_list = []
            tag_list = []
            word_list, tag_list = get_pos_tag(sent.split())
            extract_topic(word_list, tag_list)
        sorted_tags = sorted(tags.items(), key=operator.itemgetter(1), reverse=True)
        sorted_other = sorted(other.items(), key=operator.itemgetter(1), reverse=True)

        with open(os.path.join(os.pardir, "outnew", "cate", "words_{}.txt".format(cate)), "a") as out_file:
            for word, frequency in sorted_tags:
                out_file.write("{:<20}{}\n".format(word, frequency))

        with open(os.path.join(os.pardir, "outnew", "cate", "other_{}.txt".format(cate)), "a") as out_file:
            for word, frequency in sorted_other:
                out_file.write("{:<20}{}\n".format(word, frequency))

    #         word_list = sent.split()
    #         for w in word_list:
    #             if w not in not_word:
    #                 add_dict(word_dict, w)
    #     sorted_word_dict = sorted(word_dict.items(), key=operator.itemgetter(1), reverse=True)
    #     with open(os.path.join(os.pardir, "outnew", "cate", "{}.txt".format(cate)), "a") as out_file:
    #         for word, frequency in sorted_word_dict:
    #             out_file.write("{:<20}{}\n".format(word, frequency))



    #
    # sorted_jjr = sorted(jjr.items(), key=operator.itemgetter(1), reverse=True)
    # sorted_jj = sorted(jj.items(), key=operator.itemgetter(1), reverse=True)
    # sorted_nn = sorted(nn.items(), key=operator.itemgetter(1), reverse=True)
    # sorted_rbr = sorted(rbr.items(), key=operator.itemgetter(1), reverse=True)
    # sorted_other = sorted(other.items(), key=operator.itemgetter(1), reverse=True)
    #
    # with open(os.path.join(os.pardir, "outnew", "cate", "{}_jjr.txt".format(cate)), "a") as out_file:
    #     for word, frequency in sorted_jjr:
    #         out_file.write("{:<20}{}\n".format(word, frequency))
    #
    # with open(os.path.join(os.pardir, "outnew", "cate", "{}_jj.txt".format(cate)), "a") as out_file1:
    #     for word, frequency in sorted_jj:
    #         out_file1.write("{:<20}{}\n".format(word, frequency))
    #
    # with open(os.path.join(os.pardir, "outnew", "cate", "{}_nn.txt".format(cate)), "a") as out_file2:
    #     for word, frequency in sorted_nn:
    #         out_file2.write("{:<20}{}\n".format(word, frequency))
    #
    # with open(os.path.join(os.pardir, "outnew", "cate", "{}_rbr.txt".format(cate)), "a") as out_file3:
    #     for word, frequency in sorted_rbr:
    #         out_file3.write("{:<20}{}\n".format(word, frequency))
    #
    # with open(os.path.join(os.pardir, "outnew", "cate", "{}_other.txt".format(cate)), "a") as out_file4:
    #     for word, frequency in sorted_other:
    #             out_file4.write("{:<20}{}\n".format(word, frequency))


main()

print(datetime.datetime.now())

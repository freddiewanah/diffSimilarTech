import os
import pickle
import json, math
from collections import Counter
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import community
from textblob import TextBlob as tb
from nltk import pos_tag
from nltk.parse import CoreNLPParser
from datetime import datetime
import os
import nltk
import os.path
import spacy
from spacy.matcher import Matcher
import time
import gensim, os, pickle
from gensim.corpora import Dictionary
from gensim.similarities import WmdSimilarity
aspects = {}
new_aspects = {}
# Prepare tf-idf
def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def recursive_detector(G, nnodes, part, communities):
    if G.number_of_nodes() < nnodes * part:
        communities.append(G.nodes)
        return communities
    else:
        communities_generator = community.girvan_newman(G)
        temp_communities = next(communities_generator)
        # communities = sorted(map(sorted, temp_communities))
        for com in temp_communities:
            recursive_detector(G.subgraph(com), nnodes, part, communities)
        return communities


def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)


def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))


def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

query_flag = False
ver_flag = True
def set_shreshold(a, b):
    if ver_flag:
        if a == b:
            return 0.52
        return 0.55 - 0.05 ** abs(a - b)
    else:
        if a == b:
            return 0.55
        elif a > 3 or b > 3:
            return 0.55 - 0.1 ** abs(a - b)
        return 0.55 - 0.05 ** abs(a - b)

flag = False
cv = {"beat", "beats", "prefer", "prefers", "recommend", "recommends",
      "defeat", "defeats", "kill", "kills", "lead", "leads", "obliterate",
      "obliterates", "outclass", "outclasses", "outdo", "outdoes",
      "outperform", "outperforms", "outplay", "outplays", "overtake",
      "overtakes", "smack", "smacks", "subdue", "subdues", "surpass",
      "surpasses", "trump", "trumps", "win", "wins", "blow", "blows",
      "decimate", "decimates", "destroy", "destroys", "buy", "buys",
      "choose", "chooses", "favor", "favors", "grab", "grabs", "pick",
      "picks", "purchase", "purchases", "select", "selects", "race",
      "races", "compete", "competes", "match", "matches", "compare",
      "compares", "lose", "loses", "suck", "sucks"}
# Prepare POS tagger
pos_tag_set = {"JJR", "JJ", "NN", "NNS", "NNP", "NNPS", "RBR", "RBS", "JJS"}
# keywords_path = os.path.join(os.pardir, "communities", "{}.txt".format("&".join(pair)))
# stopwords_path = os.path.join(os.pardir, "communities", "stopwords.txt")
np = {"couldn", "wouldn", "shouldn", "doesn", "not", "cannot", "isn", "aren"}
# Prepare stop words set
stop_words = pickle.load(open(os.path.join(os.pardir, "data", "stop_words.pkl"), 'rb'))

nlp = spacy.load("en_core_web_sm")
# Prepare stop phrases
stop_phrases = [["for", "example"], ["in", "terms", "of"], ["keep", "in", "mind"],
                ["in", "this", "case"],
                ["a", "bit"], ["of", "course"], ["due", "to"], ["generally", "speaking"],
                ["in", "general"], ["at", "the", "moment"], ["from", "my", "point", "of", "view"],
                ["in", "my", "experience"], ["at", "least"], ["at", "most"],
                ["from", "my", "experience"], ["in", "so", "many", "ways"],
                ["hard", "data"], ["sorted", "data"], ["unsorted", "data"],
                ["by", "index"], ["new", "element"], ["are", "familiar", "of"],
                ["ios", "google-chrome"], ["several", "tests"]]

class OldPatternMatcher:

    cv = {"beat", "beats", "prefer", "prefers", "recommend", "recommends",
          "defeat", "defeats", "kill", "kills", "lead", "leads", "obliterate",
          "obliterates", "outclass", "outclasses", "outdo", "outdoes",
          "outperform", "outperforms", "outplay", "outplays", "overtake",
          "overtakes", "smack", "smacks", "subdue", "subdues", "surpass",
          "surpasses", "trump", "trumps", "win", "wins", "blow", "blows",
          "decimate", "decimates", "destroy", "destroys", "buy", "buys",
          "choose", "chooses", "favor", "favors", "grab", "grabs", "pick",
          "picks", "purchase", "purchases", "select", "selects", "race",
          "races", "compete", "competes", "match", "matches", "compare",
          "compares", "lose", "loses", "suck", "sucks"}
    cin = {"than", "over", "beyond", "upon", "as", "against", "out", "behind",
           "under", "between", "after", "unlike", "with", "by", "opposite"}

    def __init__(self):
        self.count = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0,
                      "6": 0, "7": 0, "8": 0, "9": 0, "10": 0}
        self.compa_sent_count = 0


        self.matcher = Matcher(nlp.vocab)
        self.matcher.add(0,
                    None,
                    [{'ORTH': 'JJR'}, {'ORTH': 'CIN'}, {'ORTH': 'TECH'}],
                    [{'ORTH': 'JJR'}, {}, {'ORTH': 'CIN'}, {'ORTH': 'TECH'}],
                    [{'ORTH': 'JJR'}, {'ORTH': 'CIN'}, {}, {'ORTH': 'TECH'}],
                    [{'ORTH': 'JJR'}, {}, {'ORTH': 'CIN'}, {}, {'ORTH': 'TECH'}],
                         [{'ORTH': 'JJ'}, {'ORTH': 'CIN'}, {'ORTH': 'TECH'}],
                         [{'ORTH': 'JJ'}, {}, {'ORTH': 'CIN'}, {'ORTH': 'TECH'}],
                         [{'ORTH': 'JJ'}, {'ORTH': 'CIN'}, {}, {'ORTH': 'TECH'}],
                         [{'ORTH': 'JJ'}, {}, {'ORTH': 'CIN'}, {}, {'ORTH': 'TECH'}]
                         )
        self.matcher.add(1,
                    None,
                    [{'ORTH': 'VB'}, {'ORTH': 'TECH'}, {'ORTH': 'TO'}, {'ORTH': 'VB'}],
                         [{'ORTH': 'VB'}, {'ORTH': 'TECH'}, {}, {'ORTH': 'TO'}, {'ORTH': 'VB'}],

                         )
        self.matcher.add(8,
                    None,
                    [{'ORTH': 'RBR'}, {'ORTH': 'JJ'}, {'ORTH': 'CIN'}, {'ORTH': 'TECH'}],
                    [{'ORTH': 'RBR'}, {'ORTH': 'JJ'}, {}, {'ORTH': 'CIN'}, {'ORTH': 'TECH'}])
        self.matcher.add(2,
                    None,
                    [{'ORTH': 'CV'}, {'ORTH': 'CIN'}, {'ORTH': 'TECH'}],
                    [{'ORTH': 'CV'}, {}, {'ORTH': 'CIN'}, {'ORTH': 'TECH'}])
        self.matcher.add(3,
                    None,
                    [{'ORTH': 'CV'}, {'ORTH': 'VBG'}, {'ORTH': 'TECH'}])
        self.matcher.add(5,
                         None,

                         [{'ORTH': 'TECH'}, {'ORTH': 'VB'}, {'ORTH': 'NN'}],
                         [{'ORTH': 'TECH'}, {}, {'ORTH': 'VB'}, {'ORTH': 'NN'}],
                         [{'ORTH': 'TECH'}, {'ORTH': 'VB'}, {}, {'ORTH': 'NN'}],
                         [{'ORTH': 'TECH'}, {}, {'ORTH': 'VB'}, {}, {'ORTH': 'NN'}],

                         [{'ORTH': 'TECH'}, {'ORTH': 'VB'}, {'ORTH': 'NN'}],
                         [{'ORTH': 'TECH'}, {}, {'ORTH': 'VB'}, {'ORTH': 'NN'}],
                         [{'ORTH': 'TECH'}, {'ORTH': 'VB'}, {}, {'ORTH': 'NN'}],
                         [{'ORTH': 'TECH'}, {}, {'ORTH': 'VB'}, {}, {'ORTH': 'NN'}],
                         )

        # self.matcher.add(6,
        #             None,
        #             [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {'ORTH': 'JJS'}],
        #             [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {'ORTH': 'JJS'}],
        #             [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'JJS'}],
        #             [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'JJS'}])
        self.matcher.add(7,
                    None,
                    [{'ORTH': 'TECH'}, {'ORTH': 'VB'}, {'ORTH': 'JJR'}],
                    [{'ORTH': 'TECH'}, {}, {'ORTH': 'VB'}, {'ORTH': 'JJR'}],
                    [{'ORTH': 'TECH'}, {'ORTH': 'VB'}, {}, {'ORTH': 'JJR'}],
                    [{'ORTH': 'TECH'}, {'ORTH': 'VB'}, {}, {'ORTH': 'JJR'}],
                    [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {}, {}, {'ORTH': 'JJR'}],
                    [{'ORTH': 'TECH'}, {}, {'ORTH': 'VB'}, {}, {'ORTH': 'JJR'}],
                         [{'ORTH': 'TECH'}, {'ORTH': 'VB'}, {'ORTH': 'JJR'}],
                         [{'ORTH': 'TECH'}, {}, {'ORTH': 'VB'}, {'ORTH': 'JJR'}],
                         [{'ORTH': 'TECH'}, {'ORTH': 'VB'}, {}, {'ORTH': 'JJR'}],
                         [{'ORTH': 'TECH'}, {'ORTH': 'VB'}, {}, {'ORTH': 'JJR'}],
                         [{'ORTH': 'TECH'}, {'ORTH': 'VB'}, {}, {}, {}, {'ORTH': 'JJR'}],
                         [{'ORTH': 'TECH'}, {}, {'ORTH': 'VB'}, {}, {'ORTH': 'JJR'}],
                         [{'ORTH': 'TECH'}, {}, {'ORTH': 'JJR'}]
                         )
        self.matcher.add(10,
                    None,
                    [{'ORTH': 'TECH'}, {'ORTH': 'VB'}, {'ORTH': 'RBR'}],
                    [{'ORTH': 'TECH'}, {}, {'ORTH': 'VB'}, {'ORTH': 'RBR'}],
                    [{'ORTH': 'TECH'}, {'ORTH': 'VB'}, {}, {'ORTH': 'RBR'}],
                    [{'ORTH': 'TECH'}, {'ORTH': 'VB'}, {}, {'ORTH': 'RBR'}],
                    [{'ORTH': 'TECH'}, {'ORTH': 'VB'}, {}, {}, {}, {'ORTH': 'RBR'}],
                    [{'ORTH': 'TECH'}, {}, {'ORTH': 'VB'}, {}, {'ORTH': 'RBR'}],
                    [{'ORTH': 'TECH'}, {'ORTH': 'VB'}, {'ORTH': 'RBR'}],
                    [{'ORTH': 'TECH'}, {}, {'ORTH': 'VB'}, {'ORTH': 'RBR'}],
                    [{'ORTH': 'TECH'}, {'ORTH': 'VB'}, {}, {'ORTH': 'RBR'}],
                    [{'ORTH': 'TECH'}, {'ORTH': 'VB'}, {}, {'ORTH': 'RBR'}],
                    [{'ORTH': 'TECH'}, {'ORTH': 'VB'}, {}, {}, {'ORTH': 'RBR'}],
                    [{'ORTH': 'TECH'}, {}, {'ORTH': 'VB'}, {}, {'ORTH': 'RBR'}],
                         [{'ORTH': 'TECH'}, {}, {'ORTH': 'RBR'}]
                         )
        # self.matcher.add(9,
        #             None,
        #             [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {'ORTH': 'RBS'}],
        #             [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {'ORTH': 'RBS'}],
        #             [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'RBS'}],
        #             [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'RBS'}])

        self.matcher.add(11,
                         None,

                         [{'ORTH': 'TECH'}, {'ORTH': 'NP'}],
                         [{'ORTH': 'TECH'}, {}, {'ORTH': 'NP'}],
                         [{'ORTH': 'TECH'}, {}, {'ORTH': 'NP'}],

                         )
    def add_pos_tag(self, words, tech_pair):

        tagged_words = nltk.pos_tag(words.split())
        # print(words)
        # print (tagged_words)
        tag_list = []
        for (word, tag) in tagged_words:
            if tag == "IN" and word in self.cin:
                tag_list.append("CIN")
            elif word == tech_pair.split()[0] or word == tech_pair.split()[1]:
                tag_list.append("TECH")
            elif word in np:
                tag_list.append("NP")
            elif tag[:2] == "VB" and word in cv:
                tag_list.append("CV")
            elif tag[:2] == "VB":
                tag_list.append("VB")
            elif tag[:2] == "RB":
                tag_list.append("RBR")

            else:
                tag_list.append(tag)

        return tag_list
def get_compare_words(techA, techB, sent):
    new_matcher = OldPatternMatcher()
    tag_list = new_matcher.add_pos_tag(sent, techA+' '+techB)
    patterns = new_matcher.matcher(nlp(u'{}'.format(" ".join(tag_list))))
    if patterns != []:
        p = patterns[-1]
        keywords = sent.split()[p[1]:p[2]]
        return ' '.join(keywords).replace(techA,'').replace(techB,'')

    else:
        print(sent)
        print(tag_list)
        if ';' in sent:
            sents = sent.split(';')
            reA = get_compare_words(techA, techB, sents[0])
            reB = get_compare_words(techB, techA, sents[1])
            if len(reA) > 0:
                return reA
            else:
                return reB
        return ''

new_dic = {}

# read from all sentences
def read_relation(path):
    """ Read relation files and process

        (str) -> (dict)
    """

    file_path = os.path.join(os.pardir, "outFinal", path)
    relations_file = open(file_path, 'rb')
    relations = pickle.load(relations_file)
    relations_file.close()
    return relations
all_sentences = read_relation("all_sentences.pkl")


out_path = os.path.join(os.pardir, 'data', 'web_tech_pairs.txt')
final ={}
# Read tech pairs file from data
with open(out_path, "r") as out_file:
    for line in out_file:
        tech_pairs = line[:-1].split(' ')
        techA, techB = tech_pairs
        if techA != 'dsa' or techB != 'rsa':
            continue
        sentences = []
        print("Start predict for:{}".format(techA + '_' + techB))
        if (techA, techB) in all_sentences.keys():
            sentences = all_sentences[(techA, techB)]
        elif(techB, techA) in all_sentences.keys():
            sentences = all_sentences[(techB, techA)]
        if len(sentences) == 0:
            continue
        formated_sentences = []
        for s in sentences:
            if s[1] == '':
                #do something call functionality
                keywords = get_compare_words(techA,techB,s[-1])
                s = list(s)
                s[1] = keywords
                s = tuple(s)
            flag = True
            for fs in formated_sentences:
                if s[-2]==fs[-2]:
                    sent1 = s[-1].strip('\n').split(' ')
                    sent2 = fs[-1].strip('\n').split(' ')
                    if (sent1[0] == sent2[0] and sent1[1] == sent2[1] and sent1[2] == sent2[2]) or (sent1[-1]==sent2[-1] and sent1[-2]==sent2[-2] and sent1[-3]==sent2[-3]):
                        flag = False

                        break
            if flag:
                formated_sentences.append(s)

        final[line[:-1]] = formated_sentences

print(final)
# input_file = "final1.json"
# with open(input_file, "w") as file:
#     json.dump(final, file)


# def process(all_sents, pair):
#     information = {}
#     sentences = set()
#     for items in all_sents:
#         sentences.add(items[5])
#         information[items[5]] = [items[0], items[1], items[2], items[4]]
#     sentences = list(sentences)
#     l = len(sentences)
#     corpus = []
#     topics = []
#     for sentence in sentences:
#         if True:
#             words = sentence.split()
#             words[-1] = words[-1].strip()
#             tagged_words = CoreNLPParser(url='http://localhost:9000', tagtype='pos').tag(words)
#             if len(words) != len(tagged_words):
#                 tagged_words = pos_tag(words)
#             # print(tagged_words)
#             # print(sentence.strip())
#             for phrase in stop_phrases:
#                 n = len(phrase)
#                 for i in range(len(tagged_words) - n + 1):
#                     if phrase == words[i:i + n]:
#                         for j in range(i, i + n):
#                             tagged_words[j] = (None, tagged_words[j][1])
#             i = 0
#             indices = []
#             keywords = []
#             for (word, tag) in tagged_words:
#                 if word in pair:
#                     indices.append(i)
#                     keywords.append(word)
#                     i += 1
#                 elif word not in stop_words and tag in pos_tag_set and word is not None:
#                     keywords.append(word)
#                     i += 1
#             # topics.append(" ".join(keywords))
#             # topics.append(sentence.strip())
#             if len(keywords) <= 10 and flag:
#                 ws = [w for w in keywords if w not in pair]
#             else:
#                 ws = []
#                 # if len(indices) == 2:
#                 #     for j in range(len(keywords)):
#                 #
#                 #         if j > indices[0] and j <= indices[0] + 4 and keywords[j] not in pair and j < indices[1]:
#                 #             ws.append(keywords[j])
#                 #         elif j >= indices[1] - 2 and j <= indices[1] + 2 and keywords[j] not in pair:
#                 #             ws.append(keywords[j])
#                 # else:
#                 if True:
#                     for j in range(len(keywords)):
#                         for i in indices:
#                             if j >= i - 2 and j <= i + 2 and keywords[j] not in pair and keywords[j] not in ws:
#                                 ws.append(keywords[j])
#                                 break
#             # with open(keywords_path, "a") as keywords_file:
#             #     keywords_file.write(",".join(ws)+"\n")
#             #     keywords_file.write(sentence+"\n")
#             corpus.append(ws)
#             topics.append(" ".join(ws))
#         else:
#             corpus.append([w for w in sentence.split() if w not in stop_words])
#
#     if False:
#         with open(os.path.join(os.pardir, "keywords", "corpus.pkl"), 'wb') as corpus_file:
#             pickle.dump(corpus, corpus_file)
#         with open(os.path.join(os.pardir, "keywords", "sentences.pkl"), 'wb') as sentences_file:
#             pickle.dump(sentences, sentences_file)
#
#     else:
#         # Prepare word2vector model
#         fname = os.path.join(os.pardir, "data", "mynewmodel")
#         model = gensim.models.Word2Vec.load(fname)
#         model.init_sims(replace=True)
#
#         # Build weighted graph
#         # dictionary = Dictionary(corpus)
#         # bow_corpus = [dictionary.doc2bow(document) for document in corpus]
#         index = WmdSimilarity(corpus, model)
#
#         G = nx.Graph()
#         for i in range(l - 1):
#             sims = index[corpus[i]]
#             # print("query:")
#             # print(corpus[i])
#             # print(sentences[i])
#             # print("sims:")
#             for j in range(i + 1, l):
#                 # print(sims[j])
#                 # print(corpus[j])
#                 # print(sentences[j])
#                 # print()
#                 shreshold = set_shreshold(len(corpus[i]), len(corpus[j]))
#                 if sims[j] >= shreshold:
#                     if i not in G: G.add_node(i)
#                     if j not in G: G.add_node(j)
#                     G.add_edge(i, j)
#                     # G.add_edge(i, j, weight=sims[j])
#
#         out_path = os.path.join(os.pardir, "recursive_communities",
#                                 "{}_{}_{}.txt".format("&".join(pair), G.number_of_nodes(), l))
#         # image_path = os.path.join(os.pardir, com_dir, "{}_{}_{}.png".format("&".join(pair), G.number_of_nodes(), l))
#
#         # Draw graph
#         pos = nx.spring_layout(G)
#         plt.figure(figsize=(19, 12))
#         plt.axis('off')
#         nx.draw_networkx_nodes(G, pos, node_size=50)
#         nx.draw_networkx_edges(G, pos, width=0.75)
#         # plt.savefig(image_path)
#         # plt.show()
#
#         nnodes = G.number_of_nodes()
#         if nnodes < 4:
#             communities = []
#             communities.append(G.nodes())
#             # return
#         elif nnodes <= 15:
#             communities_generator = community.girvan_newman(G)
#             temp_communities = next(communities_generator)
#             communities = sorted(map(sorted, temp_communities))
#             # return
#         else:
#             if nnodes < 70:
#                 part = 2 / 3
#             else:
#                 part = 1 / 3
#             # Detect communities
#             communities = recursive_detector(G, nnodes, part, [])
#         num = 0
#         graph_indices = set()
#         bloblist = []
#         clusters = []
#         ss = []
#         for com in communities:
#             if len(com) > 1:
#                 doc = ""
#                 for i in com:
#                     doc += topics[i] + " "
#                 bloblist.append(tb(doc))
#                 clusters.append(com)
#         # print(bloblist)
#         # print(clusters)
#         # print(sentences)
#         # print(information)
#         aspects[pair] = set()
#         new_aspects[pair] = {}
#         for i, blob in enumerate(bloblist):
#             scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
#             sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
#             aspect_keywords = []
#             # print(sorted_words)
#             if len(sorted_words) > 3:
#                 keywords =  ' '.join([i[0] for i in sorted_words[:3]])
#             else:
#                 keywords = ' '.join(i[0] for i in sorted_words)
#             for j in clusters[i]:
#                 if len(information[sentences[j]][1]) == 0:
#                     information[sentences[j]][1] = 'general'
#                 information[sentences[j]][1] = information[sentences[j]][1]+' in '+keywords
#                 graph_indices.add(j)
#
#         for j in range(len(sentences)):
#             if j not in graph_indices:
#                 if len(information[sentences[j]][1]) == 0:
#                     information[sentences[j]][1] = 'general'
#                 information[sentences[j]][1] = information[sentences[j]][1] + ' overall'
#         return information
#
#
#
# input_file = "final1.json"
# output_file = "test1.csv"
# with open(input_file, 'r') as outfile:
#     with open('csvfile1.csv', 'w+') as file:
#         final = json.load(outfile)
#         for key in final.keys():
#             print(key)
#             # print(final[key])
#             techs = key.split(' ')
#             info = process(final[key], (techs[0], techs[1]))
#             print(info)
#             limit = 10
#             quality = []
#             id = []
#             examples = []
#             i = 0
#             for sent in info.keys():
#                 if i > 10:
#                     # write to files
#                     file.write('\"{}\",\"{}\",\"{}\",\"{}\",\"{}\"\n'.format(techs[0],techs[1], ','.join(quality)+',', ','.join(id)+',', ','.join(examples)+','))
#                     quality = []
#                     id = []
#                     examples = []
#                     i = 0
#                 quality.append(info[sent][1])
#                 examples.append(sent.replace('\n',''))
#                 id.append(info[sent][-1])
#                 i += 1
#             file.write('\"{}\",\"{}\",\"{}\",\"{}\",\"{}\"\n'.format(techs[0], techs[1], ','.join(quality) + ',',
#                                                                      ','.join(id) + ',', ','.join(examples) + ','))
#
#
#
#
#


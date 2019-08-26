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

ver_flag= False
print(datetime.datetime.now())
# Prepare sentences
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
print(stackoverflow.keys)

sentences = []
flag = True  # all sentences
# Prepare stop words set
stop_words = pickle.load(open(os.path.join(os.pardir, "data", "stop_words.pkl"), 'rb'))
# Prepare POS tagger
pos_tag_set = {"JJR", "JJ", "NN", "NNS", "NNP", "NNPS", "RBR", "RBS", "JJS"}
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

pos_flag = False

# Process sentences
for pair in stackoverflow:
    cp = [sentence[-1] for sentence in stackoverflow[pair]]
    for s in cp:
        s = s.replace(pair[0], "B")
        s = s.replace(pair[1], "B")
        if s not in sentences:
            sentences.append(s)

l = len(sentences)
corpus = []
topics = []
for sentence in sentences:
    if pos_flag:
        words = sentence.split()
        words[-1] = words[-1].strip()
        tagged_words = CoreNLPParser(url='http://localhost:9000', tagtype='pos').tag(words)
        if len(words) != len(tagged_words):
            tagged_words = pos_tag(words)
        # print(tagged_words)
        # print(sentence.strip())
        for phrase in stop_phrases:
            n = len(phrase)
            for i in range(len(tagged_words) - n + 1):
                if phrase == words[i:i+n]:
                    for j in range(i, i+n):
                        tagged_words[j] = (None, tagged_words[j][1])
        i = 0
        indices = []
        keywords = []
        for (word, tag) in tagged_words:
            if word in pair:
                indices.append(i)
                keywords.append(word)
                i += 1
            elif word not in stop_words and tag in pos_tag_set and word is not None:
                keywords.append(word)
                i += 1
        # topics.append(" ".join(keywords))
        # topics.append(sentence.strip())
        if len(keywords) <= 10 and flag:
            ws = [w for w in keywords if w != 'A' and w != 'B']
        else:
            ws = []
            # if len(indices) == 2:
            #     for j in range(len(keywords)):
            #
            #         if j > indices[0] and j <= indices[0] + 4 and keywords[j] not in pair and j < indices[1]:
            #             ws.append(keywords[j])
            #         elif j >= indices[1] - 2 and j <= indices[1] + 2 and keywords[j] not in pair:
            #             ws.append(keywords[j])
            # else:
            if True:
                for j in range(len(keywords)):
                    for i in indices:
                        if j >= i - 2 and j <= i + 2 and keywords[j] not in pair and keywords[j] not in ws:
                            ws.append(keywords[j])
                            break
        # with open(keywords_path, "a") as keywords_file:
        #     keywords_file.write(",".join(ws)+"\n")
        #     keywords_file.write(sentence+"\n")
        corpus.append(ws)
        topics.append(" ".join(ws))
    else:
        corpus.append([w for w in sentence.split() if w not in stop_words])

# with open(os.path.join(os.pardir, "outnew", "corpus.pkl"), 'wb') as corpus_file:
#     pickle.dump(corpus, corpus_file)
# with open(os.path.join(os.pardir, "outnew", "sentences.pkl"), 'wb') as sentences_file:
#     pickle.dump(sentences, sentences_file)

print(datetime.datetime.now())
print("finished gathering sentences")
# Prepare word2vector model
# model = gensim.models.Word2Vec(sentences, min_count=20, size=200, workers=8)
print(1)
corpus_formated = []
original_words = []
print(len(corpus))
for c in corpus:
    temp = c

    if len(c) > 0 and len(c)<=25:
        corpus_formated.append(' '.join(c))
        original_words.append(' '.join(temp))
# print(corpus_formated)
print(len(corpus_formated))
print(corpus_formated)
# bc = BertClient()
# print(bc.encode(['in short A is the B of flex it works but will teach you a lot of bad habits',
#                  'A means that your B is no longer com-visible']))
# test_text = ['in short A is the B of flex it works but will teach you a lot of bad habits',
#                   'A means that your B is no longer com-visible']
# vec = bc.encode(corpus_formated)

# np.save('bert_vec_unfiltered', vec)
vec = np.load('bert_vec_unfiltered.npy')

print(2)
plt.figure(figsize=(10, 7))

dendrogram = sch.dendrogram(sch.linkage(vec, method='ward'))
# plt.show()

cluster = AgglomerativeClustering(n_clusters=38, affinity='euclidean', linkage='ward')
clusters = cluster.fit_predict(vec)

plt.figure(figsize=(10, 7))
plt.scatter(vec[:,0], vec[:,1], c=cluster.labels_, cmap='rainbow')

for index, val in enumerate(clusters):
    out_file = open(os.path.join(os.pardir, "outnew", "bert_unfiltered", "{}.txt".format(val)), "a")
    out_file.write(original_words[index])
    out_file.write("\n")
    out_file.close()
print(clusters)
plt.show()

print(datetime.datetime.now())

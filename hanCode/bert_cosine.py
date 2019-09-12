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
from sklearn.cluster import KMeans
import gensim, os, pickle
from gensim.corpora import Dictionary
from gensim.similarities import WmdSimilarity
import math
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import community
import nltk
from nltk import pos_tag
import tensorflow as tsf
import tensorflow_hub as hub
from textblob import TextBlob as tb
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

ver_flag= False
print(datetime.datetime.now())

nltk.download('punkt')
# Prepare tf-idf
def tf(word, blob):
    return blob.words.count(word) / len(blob.words)


def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)


def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))


def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)


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


#
# # Process sentences
# for pair in stackoverflow:
#     cp = [sentence[-1] for sentence in stackoverflow[pair]]
#     for s in cp:
#         s = s.replace(pair[0], "B")
#         s = s.replace(pair[1], "B")
#         if s not in sentences:
#             sentences.append(s)
#
# l = len(sentences)

pair = ("compiled-language","interpreted-language")

vec_sentences = ["a compiled-language will generally run faster than an interpreted-language so i think ruby and php start behind the eight ball but it really comes down to how you use the language and how you structure the code", "an interpreted-language will typically run one to two orders of magnitude slower than a compiled-language", "and perl like any interpreted-language is much slower than a compiled-language", "are compiled-language better than interpreted-language or vice-versa", "fact is that interpreted-language like php are always slower than a compiled-language", "from what i know a compiled-language such as c++ is much faster than an interpreted-language such as javascript", "in my general programming experience compiled c c++ programs generally run faster than most other compiled-language like java or even compiled python and almost always run faster than interpreted-language like uncompiled python or javascript", "interpreted-language execution speed are slower than compiled-language true but once there is need for more speed you can call in compiled stuff through gems or micro services", "interpreted-language tend to be but not always are significantly slower than compiled-language", "it should be noted that interpreted-language are inherently many time slower than natively compiled-language", "mostly interpreted-language are a bit slower compared with compiled-language but i guess the difference is almost negligible in coffeescript javascript because of node.js", "naturally interpreted-language will run slower than compiled-language as compiled code can be ran blindly by the cpu where as compiled code needs to be checked ran line by line", "php is an interpreted-language so will run a little slower than a compiled-language", "python is an interpreted-language so by definition is slower than other compiled-language but the drawback in the execution speed is not even noticeable in most of applications", "that being said a compiled-language like c will almost always be faster than an interpreted-language like javascript", "this is a good question but should be formulated a little different in my opinion for example why are interpreted-language slower than compiled-language", "this makes interpreted-language generally slower than compiled-language due to the overhead of running the vm or interpreter", "while ruby and python are both interpreted-language and operation-for-operation slower than compiled-language the reality is in executing an application only a small portion of cpu time is spent in your code and the majority is spent into the built-in libraries you call into which are often native implementations and thus are as fast as compiled code", "performance of programs in compiled-language is significantly better than that of an interpreted-language", "writing in a compiled-language java or c++ in your examples would almost certainly give better performance than an interpreted-language like php", "while java could be described as a compiled and interpreted-language it s probably easier to think of java itself as a compiled-language and java bytecode as an interpreted-language", "an interpreted-language surely makes it easier but this is still entirely possible with compiled-language like c", "interpreted-language are inherently less performant than compiled-language - c will generally outperform python - some operations more than others", "then c which is one those languages closer to the processor level is very performant and generally speaking compiled-language because they turn your code into assembly language are more performant than interpreted-language", "my guess is that in interpreted-language the efficiency benefit in using switch statements is indeed smaller than in compiled-language", "this is usually seen in dynamic interpreted-language but is less common in compiled-language", "especially in an interpreted-language like php where classes add more overhead than a compiled-language"]


bc = BertClient(check_length=False)

# Prepare sentences
in_path = os.path.join(os.pardir, "data", "relations.pkl")
relations_file = open(in_path, 'rb')
relations = pickle.load(relations_file)
relations_file.close()
def main():
    information = {}
    sentences = set()
    for items in relations[pair]:
        sentences.add(items[5])
        information[items[5]] = (items[0], items[1], items[2], items[4])
    aspects = {}
    new_aspects = {}
    G = nx.Graph()
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
    print(1)

    print(2)
    init_op = tsf.global_variables_initializer()
    init_table = tsf.tables_initializer()
    sess = tsf.Session()
    sess.run(init_op)
    sess.run(init_table)
    corpus=[]
    vec_sentences_formatted=[]
    for idx in range(len(vec_sentences)):
        temp = vec_sentences[idx].replace(pair[0], "*")
        temp = temp.replace(pair[0], "*")
        vec_sentences_formatted.append(temp)
        corpus.append(' '.join([w for w in temp.split() if w not in stop_words]))
    vec = sess.run(embed(vec_sentences_formatted))
    print(datetime.datetime.now())
    print(vec_sentences_formatted)
    l = len(vec_sentences_formatted)
    sim_metrics = [[0 for i in range(l)] for j in range(l)]
    for i in range(l - 1):

        # print("query:")
        # print(corpus[i])
        # print(sentences[i])
        # print("sims:")
        for j in range(i + 1, l):
            sims = cosine_sim(vec[i], vec[j])
            sim_metrics[i][j] = sims
            sim_metrics[j][i] = sims
            # print(sims[j])
            # print(corpus[j])
            # print(sentences[j])
            # print()
            # shreshold = set_shreshold(len(corpus[i]), len(corpus[j]))
            if sims >= 0.7:
                if i not in G: G.add_node(i)
                if j not in G: G.add_node(j)
                G.add_edge(i, j)
                # G.add_edge(i, j, weight=sims[j])

    out_path = os.path.join(os.pardir, "communities", "{}_{}_{}.txt".format("&".join(pair), G.number_of_nodes(), l))
    # image_path = os.path.join(os.pardir, com_dir, "{}_{}_{}.png".format("&".join(pair), G.number_of_nodes(), l))
    print(sim_metrics)
    # Draw graph
    pos = nx.spring_layout(G)
    plt.figure(figsize=(19,12))
    plt.axis('off')
    nx.draw_networkx_nodes(G, pos, node_size=50)
    nx.draw_networkx_edges(G, pos, width=0.75)
    # plt.savefig(image_path)
    # plt.show()

    nnodes = G.number_of_nodes()

    if nnodes < 4:
        communities = []
        communities.append(G.nodes())
        return
    elif nnodes <= 15:
        communities_generator = community.girvan_newman(G)
        temp_communities = next(communities_generator)
        communities = sorted(map(sorted, temp_communities))
        return
    else:
        if nnodes < 50:
            part = 2 / 3
        else:
            part = 1 / 3
        # Detect communities
        communities = recursive_detector(G, nnodes, part, [])
    num = 0
    graph_indices = set()
    bloblist = []
    clusters = []
    for com in communities:
        if len(com) > 1:
            doc = ""
            for i in com:
                doc += "test" + " "
            bloblist.append(tb(doc))
            clusters.append(com)

    aspects[pair] = set()
    new_aspects[pair] = {}
    # if True:
    with open(out_path, "a") as out_file:
        for i, blob in enumerate(bloblist):
            # print("Top words in document {}".format(i + 1))
            scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            # word_num = 0
            aspect_keywords = []
            for word, score in sorted_words[:3]:
                out_file.write(word+", ")
                aspect_keywords.append(word)
            new_aspects[pair][" ".join(aspect_keywords)] = set()
            # for word, score in sorted_words:
            #     if word_num == 3:
            #         break
            #     if tf(word, blob) >= 0.2:
            #         word_num += 1
            #         out_file.write(word+", ")
            #         print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
            out_file.write("---------------------------------------------------\n\n")
            for j in clusters[i]:
                temp = information[vec_sentences[j]+'\n']
                new_aspects[pair][" ".join(aspect_keywords)].add((temp[0], temp[1], temp[2], temp[3], vec_sentences[j]))
                aspects[pair].add((temp[0], temp[1], temp[2], " ".join(aspect_keywords), temp[3], vec_sentences[j]))
                # out_file.write(",".join(sentences[j])+"\n")
                out_file.write(vec_sentences[j]+"\n")
                graph_indices.add(j)
            num += 1
        out_file.write("other---------------------------------------------------\n\n")
        new_aspects[pair]["other"] = set()
        for j in range(len(vec_sentences)):
            if j not in graph_indices:
                temp = information[vec_sentences[j]+'\n']
                new_aspects[pair]["other"].add((temp[0], temp[1], temp[2], temp[3], vec_sentences[j]))
                aspects[pair].add((temp[0], temp[1], temp[2], "", temp[3], vec_sentences[j]))

                # out_file.write(",".join(sentences[j])+"\n")
                out_file.write(vec_sentences[j]+"\n")
    plt.close('all')
    print(pair)


main()


#
# """
# This is a simple application for sentence embeddings: clustering
# Sentences are mapped to sentence embeddings and then k-mean clustering is applied.
# """
# from sentence_transformers import SentenceTransformer
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.cluster import KMeans
# import pickle
# import os, datetime
# import numpy as np
# import math
# import matplotlib.pyplot as plt
# import networkx as nx
# from networkx.algorithms import community
# from textblob import TextBlob as tb
# import nltk
# nltk.download('punkt')
# embedder = SentenceTransformer('bert-large-nli-stsb-mean-tokens')
# def cosine_sim(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
#
# ver_flag= False
# print(datetime.datetime.now())
#
# # Prepare tf-idf
# def tf(word, blob):
#     return blob.words.count(word) / len(blob.words)
#
#
# def n_containing(word, bloblist):
#     return sum(1 for blob in bloblist if word in blob.words)
#
#
# def idf(word, bloblist):
#     return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))
#
#
# def tfidf(word, blob, bloblist):
#     return tf(word, blob) * idf(word, bloblist)
#
#
# def set_shreshold(a, b):
#     if ver_flag:
#         if a == b:
#             return 0.52
#         return 0.55 - 0.05 ** abs(a - b)
#     else:
#         if a == b:
#             return 0.55
#         elif a > 3 or b > 3:
#             return 0.55 - 0.1 ** abs(a - b)
#         return 0.55 - 0.05 ** abs(a - b)
#
#
# def recursive_detector(G, nnodes, part, communities):
#     if G.number_of_nodes() < nnodes * part:
#         communities.append(G.nodes)
#         return communities
#     else:
#         communities_generator = community.girvan_newman(G)
#         temp_communities = next(communities_generator)
#         # communities = sorted(map(sorted, temp_communities))
#         for com in temp_communities:
#             recursive_detector(G.subgraph(com), nnodes, part, communities)
#         return communities
#
# # read from all sentences
# def read_relation(path):
#     """ Read relation files and process
#
#         (str) -> (dict)
#     """
#
#     file_path = os.path.join(os.pardir, "outFinal", path)
#     relations_file = open(file_path, 'rb')
#     relations = pickle.load(relations_file)
#     relations_file.close()
#     return relations
#
# all_sentences = read_relation("all_sentences.pkl")
#
# def main():
#     pair = ("rsa", "aes")
#     setPair = ("rsa", "aes")
#     information = {}
#     sentences = set()
#     details = list(all_sentences[pair])
#     for items in details:
#         te = items[-1].replace("\n", "")
#         sentences.add(te)
#         information[te] = (items[0], items[1], items[2], items[4])
#     sentences = list(sentences)
#     pair = list(pair)
#     corpus = []
#     for idx in range(len(sentences)):
#         temp = sentences[idx].replace(pair[0], "technology")
#         temp = temp.replace(pair[1], "technology")
#         temp = temp.replace("\n", "")
#         corpus.append(temp)
#     aspects = {}
#     new_aspects = {}
#     corpus_embeddings = embedder.encode(corpus)
#     G = nx.Graph()
#     l = len(corpus)
#     sim_metrics = [[0 for i in range(l)] for j in range(l)]
#     for i in range(l - 1):
#
#         # print("query:")
#         # print(corpus[i])
#         # print(sentences[i])
#         # print("sims:")
#         for j in range(i + 1, l):
#             sims = cosine_sim(corpus_embeddings[i], corpus_embeddings[j])
#             sim_metrics[i][j] = sims
#             sim_metrics[j][i] = sims
#             # print(sims[j])
#             # print(corpus[j])
#             # print(sentences[j])
#             # print()
#             # shreshold = set_shreshold(len(corpus[i]), len(corpus[j]))
#             if sims >= 0.6:
#                 if i not in G: G.add_node(i)
#                 if j not in G: G.add_node(j)
#                 G.add_edge(i, j)
#                 # G.add_edge(i, j, weight=sims[j])
#
#     out_path = os.path.join(os.pardir, "communities", "{}_{}_{}.txt".format("&".join(pair), G.number_of_nodes(), l))
#     # image_path = os.path.join(os.pardir, com_dir, "{}_{}_{}.png".format("&".join(pair), G.number_of_nodes(), l))
#     print(sim_metrics)
#     # Draw graph
#     pos = nx.spring_layout(G)
#     plt.figure(figsize=(19,12))
#     plt.axis('off')
#     nx.draw_networkx_nodes(G, pos, node_size=50)
#     nx.draw_networkx_edges(G, pos, width=0.75)
#     # plt.savefig(image_path)
#     # plt.show()
#
#     nnodes = G.number_of_nodes()
#
#     if nnodes < 4:
#         communities = []
#         communities.append(G.nodes())
#         return
#     elif nnodes <= 15:
#         communities_generator = community.girvan_newman(G)
#         temp_communities = next(communities_generator)
#         communities = sorted(map(sorted, temp_communities))
#         return
#     else:
#         if nnodes < 50:
#             part = 2 / 3
#         else:
#             part = 1 / 3
#         # Detect communities
#         communities = recursive_detector(G, nnodes, part, [])
#     num = 0
#     graph_indices = set()
#     bloblist = []
#     clusters = []
#     for com in communities:
#         if len(com) > 1:
#             doc = ""
#             for i in com:
#                 doc += "test" + " "
#             bloblist.append(tb(doc))
#             clusters.append(com)
#
#     aspects[setPair] = set()
#     new_aspects[setPair] = {}
#     # if True:
#     with open(out_path, "a") as out_file:
#         for i, blob in enumerate(bloblist):
#             # print("Top words in document {}".format(i + 1))
#             scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
#             sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
#             # word_num = 0
#             aspect_keywords = []
#             for word, score in sorted_words[:3]:
#                 out_file.write(word+", ")
#                 aspect_keywords.append(word)
#             new_aspects[setPair][" ".join(aspect_keywords)] = set()
#             # for word, score in sorted_words:
#             #     if word_num == 3:
#             #         break
#             #     if tf(word, blob) >= 0.2:
#             #         word_num += 1
#             #         out_file.write(word+", ")
#             #         print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
#             out_file.write("---------------------------------------------------\n\n")
#             for j in clusters[i]:
#                 temp = information[sentences[j]]
#                 new_aspects[setPair][" ".join(aspect_keywords)].add((temp[0], temp[1], temp[2], temp[3], sentences[j]))
#                 aspects[setPair].add((temp[0], temp[1], temp[2], " ".join(aspect_keywords), temp[3], sentences[j]))
#                 # out_file.write(",".join(sentences[j])+"\n")
#                 out_file.write(sentences[j]+"\n")
#                 graph_indices.add(j)
#             num += 1
#         out_file.write("other---------------------------------------------------\n\n")
#         new_aspects[setPair]["other"] = set()
#         for j in range(len(sentences)):
#             if j not in graph_indices:
#                 temp = information[sentences[j]]
#                 new_aspects[setPair]["other"].add((temp[0], temp[1], temp[2], temp[3], sentences[j]))
#                 aspects[setPair].add((temp[0], temp[1], temp[2], "", temp[3], sentences[j]))
#
#                 # out_file.write(",".join(sentences[j])+"\n")
#                 out_file.write(corpus[j]+"\n")
#     plt.close('all')
#     print(pair)
#
# # # Perform kmean clustering
# # num_clusters = 3
# # clustering_model = AgglomerativeClustering(n_clusters=num_clusters,affinity='manhattan', linkage='complete')
# # clustering_model.fit(corpus_embeddings)
# # cluster_assignment = clustering_model.labels_
# #
# # clustered_sentences = [[] for i in range(num_clusters)]
# # for sentence_id, cluster_id in enumerate(cluster_assignment):
# #     clustered_sentences[cluster_id].append(sentences[sentence_id])
# #     print(cluster_id)
# #
# #
# # for i, cluster in enumerate(clustered_sentences):
# #     print("Cluster ", i+1)
# #     print(cluster)
# #     print("")
# #
# # for i, cluster in enumerate(clustered_sentences):
# #     for s in cluster:
# #         print(s.rstrip())
#
# main()
from allennlp.commands.elmo import ElmoEmbedder

elmo = ElmoEmbedder()

"""
This is a simple application for sentence embeddings: clustering
Sentences are mapped to sentence embeddings and then k-mean clustering is applied.
"""

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import pickle
import os



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

details = list(all_sentences[("rsa", "aes")])
sentences = []
for d in details:
    te = d[-1].replace("\n", "")
    if te not in sentences:
        sentences.append(te)
sentences = list(set(sentences))

pair = ["rsa", "aes"]
corpus = []
for idx in range(len(sentences)):
    temp = sentences[idx].replace(pair[0], "tech")
    temp = temp.replace(pair[1], "tech")
    temp = temp.replace("\n", "")
    if temp not in corpus:
        corpus.append(temp.split(" "))

corpus_embeddings = elmo.embed_sentences(corpus)
count = 1
final_corpus =[]
for c in corpus_embeddings:
    final_corpus.append(c[-1])


print(final_corpus[0])
# Perform kmean clustering
# num_clusters = 5
# clustering_model = AgglomerativeClustering(n_clusters=num_clusters)
# clustering_model.fit(final_corpus)
# cluster_assignment = clustering_model.labels_
#
# clustered_sentences = [[] for i in range(num_clusters)]
# for sentence_id, cluster_id in enumerate(cluster_assignment):
#     clustered_sentences[cluster_id].append(sentences[sentence_id])
#     print(cluster_id)
#
#
# for i, cluster in enumerate(clustered_sentences):
#     print("Cluster ", i+1)
#     print(cluster)
#     print("")
#
# for i, cluster in enumerate(clustered_sentences):
#     for s in cluster:
#         print(s.rstrip())
#
#
# cluster = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')
# clusterss = cluster.fit_predict(final_corpus)
# print(clusterss)
#

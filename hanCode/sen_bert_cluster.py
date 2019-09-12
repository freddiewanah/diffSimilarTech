"""
This is a simple application for sentence embeddings: clustering
Sentences are mapped to sentence embeddings and then k-mean clustering is applied.
"""
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import pickle
import os

embedder = SentenceTransformer('bert-large-nli-stsb-mean-tokens')
ps = [("google-chrome", "safari"),("compiled-language", "interpreted-language"),("sortedlist", "sorteddictionary"),
      ("heapsort","quicksort"),("ant","maven"),("pypy","cpython"),("quicksort", "mergesort"),("lxml","beautifulsoup"),
      ("awt", "swing"),("jackson","gson"),("swift", "objective-c"),("memmove","memcpy")]
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

details = list(all_sentences[("memmove","memcpy")])
sentences = []
for d in details:
    te = d[-1].replace("\n", "")
    if te not in sentences:
        sentences.append(te)
sentences = list(set(sentences))

pair = ["memmove","memcpy"]
corpus = []
for idx in range(len(sentences)):
    temp = sentences[idx].replace(pair[0], "")
    temp = temp.replace(pair[1], "")
    temp = temp.replace("\n", "")
    if temp not in corpus:
        corpus.append(temp)

corpus_embeddings = embedder.encode(corpus)
print(corpus_embeddings[0])
# Perform kmean clustering
num_clusters = 4
clustering_model = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='average')
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(sentences[sentence_id])
    print(cluster_id)

out_path = os.path.join(os.pardir, "communities", "{}.txt".format("&".join(pair)))
with open(out_path, "a") as f:
    for i, cluster in enumerate(clustered_sentences):
        f.write("Cluster "+str(i+1)+"\n")
        for cl in cluster:
            f.write(str(cl)+"\n")
        f.write("\n")

for i, cluster in enumerate(clustered_sentences):
    for s in cluster:
        print(s.rstrip())

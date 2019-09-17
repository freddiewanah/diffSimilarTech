from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import pickle
import os
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

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

sentences = []
for key in all_sentences.keys():
    for d in all_sentences[key]:
        te = d[-1].replace("\n", "")

        sentences.append(te)


sentences = list(set(sentences))
final_sentences = [s.split(" ") for s in sentences]

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(final_sentences)]

model = Doc2Vec(documents, size=5, window=2, min_count=1, workers=4)

fname = os.path.join(os.pardir,"outnew","doc2vecModel")
model.save(fname)



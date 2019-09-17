import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn import svm
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import pickle
import re,string, os
import gensim.models as g
from gensim.models import Doc2Vec
import gensim


sentences = """
while ruby and python are both interpreted-language and operation-for-operation slower than compiled-language the reality is in executing an application only a small portion of cpu time is spent in your code and the majority is spent into the built-in libraries you call into which are often native implementations and thus are as fast as compiled code
interpreted-language execution speed are slower than compiled-language true but once there is need for more speed you can call in compiled stuff through gems or micro services
an interpreted-language must do this same translation at the same time it s trying to run the program;that typically slows it down though modern interpreters and virtual machines like to brag about how they can do a few things faster because they have the extra information that a compiled-language doesn t
fact is that interpreted-language like php are always slower than a compiled-language
naturally interpreted-language will run slower than compiled-language as compiled code can be ran blindly by the cpu where as compiled code needs to be checked ran line by line
an interpreted-language will typically run one to two orders of magnitude slower than a compiled-language
mostly interpreted-language are a bit slower compared with compiled-language but i guess the difference is almost negligible in coffeescript javascript because of node.js
interpreted-language tend to be but not always are significantly slower than compiled-language
php is an interpreted-language so will run a little slower than a compiled-language
and perl like any interpreted-language is much slower than a compiled-language
it should be noted that interpreted-language are inherently many time slower than natively compiled-language
only interpreted-language can execute code at runtime;compiled-language cannot
my guess is that in interpreted-language the efficiency benefit in using switch statements is indeed smaller than in compiled-language
this is usually seen in dynamic interpreted-language but is less common in compiled-language
this is a good question but should be formulated a little different in my opinion for example why are interpreted-language slower than compiled-language
interpreted-language are inherently less performant than compiled-language - c will generally outperform python - some operations more than others
python is an interpreted-language so by definition is slower than other compiled-language but the drawback in the execution speed is not even noticeable in most of applications
a compiled-language will generally run faster than an interpreted-language so i think ruby and php start behind the eight ball but it really comes down to how you use the language and how you structure the code
this makes interpreted-language generally slower than compiled-language due to the overhead of running the vm or interpreter
java bytecode as an interpreted-language
are compiled-language better than interpreted-language or vice-versa
performance of programs in compiled-language is significantly better than that of an interpreted-language
writing in a compiled-language java or c++ in your examples would almost certainly give better performance than an interpreted-language like php
especially in an interpreted-language like php where classes add more overhead than a compiled-language
an interpreted-language surely makes it easier but this is still entirely possible with compiled-language like c
that being said a compiled-language like c will almost always be faster than an interpreted-language like javascript
in my general programming experience compiled c c++ programs generally run faster than most other compiled-language like java or even compiled python and almost always run faster than interpreted-language like uncompiled python or javascript
then c which is one those languages closer to the processor level is very performant and generally speaking compiled-language because they turn your code into assembly language are more performant than interpreted-language
from what i know a compiled-language such as c++ is much faster than an interpreted-language such as javascript
in c# this won t be so easy because c# is a compiled-language not an interpreted-language;in an interpreted-language it s fairly easy to parse raw-text as the language and see the results
then c which is one those languages closer to the processor level is very performant and generally speaking compiled-language because compiled-language turn your code into assembly language are more performant than interpreted-language
""".split("\n")[1:-1]


def preprocessing(line):
    line = line.lower()
    line = re.sub(r"[{}]".format(string.punctuation), " ", line)
    return line

tfidf_vectorizer = TfidfVectorizer(preprocessor=preprocessing)
tfidf = tfidf_vectorizer.fit_transform(sentences)


# Perform kmean clustering
num_clusters = 3
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(tfidf)
cluster_assignment = clustering_model.labels_

clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(sentences[sentence_id])
    print(cluster_id)


for i, cluster in enumerate(clustered_sentences):
    for s in cluster:
        print(s.rstrip())


LabeledSentence1 = gensim.models.doc2vec.TaggedDocument


model = fname = os.path.join(os.pardir,"outnew","doc2vecModel")
m = g.Doc2Vec.load(model)
test_sent = [ x.strip().split() for x in sentences]
X = []
for d in test_sent:
    X.append(m.infer_vector(d, alpha=0.025, steps=1000))

clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(X)
cluster_assignment = clustering_model.labels_

clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(sentences[sentence_id])
    print(cluster_id)
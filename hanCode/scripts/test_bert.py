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
import json

file = open(os.path.join(os.pardir, "outnew", "cate", "bert_category.json"), 'r')
cates = json.load(file)

bc = BertClient()

doc_vecs = np.load('../outnew/bert_cate_v4/algorithm_vec.npy')

while True:
    query = input('your question: ')
    query_vec = bc.encode([query])[0]
    print(query_vec)
    print(bc.encode([query]))
    # compute normalized dot product as score
    score = np.sum(query_vec * doc_vecs, axis=1) / np.linalg.norm(doc_vecs, axis=1)
    topk_idx = np.argsort(score)[::-1][:10]
    for idx in topk_idx:
        print('> %s\t%s' % (score[idx], cates['algorithm'][idx]))

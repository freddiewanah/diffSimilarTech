from sklearn import metrics
import os
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile


ps = ["compiled&interpreted","sortedlist&sorteddictionary",
      "heapsort&quicksort","pypy&cpython","lxml&beautifulsoup",
      "awt&swing","memmove&memcpy","jruby&mri"]

in_path = os.path.join(os.pardir, "experiment", "clusters_bert.xlsx")

for i in ps:
      df = pd.read_excel(in_path, sheet_name= i)
      print(i)
      # a = df['our model']
      # b = df['ground truth']
      # a = list(a)
      # print(type(a))
      # print(a)

      labels_true = df['truth']
      labels_pred = df['model']
      labels_rfidf = df['tf-idf']
      labels_v2d = df['doc2vec']


      preds = (("model", labels_pred))




      print("ARI	COM	FMI	HOM	NMI	V-M")

      # print(
      #       metrics.adjusted_rand_score(labels_true, labels_pred),
      #       metrics.completeness_score(labels_true, labels_pred),
      #       metrics.fowlkes_mallows_score(labels_true, labels_pred),
      #       metrics.homogeneity_score(labels_true, labels_pred),
      #       metrics.normalized_mutual_info_score(labels_true, labels_pred),
      #       metrics.v_measure_score(labels_true, labels_pred)
      # )

      # print(
      #       metrics.adjusted_rand_score(labels_true, labels_rfidf),
      #       metrics.completeness_score(labels_true, labels_rfidf),
      #       metrics.fowlkes_mallows_score(labels_true, labels_rfidf),
      #       metrics.homogeneity_score(labels_true, labels_rfidf),
      #       metrics.normalized_mutual_info_score(labels_true, labels_rfidf),
      #       metrics.v_measure_score(labels_true, labels_rfidf)
      # )

      print(
            metrics.adjusted_rand_score(labels_true, labels_v2d),
            metrics.completeness_score(labels_true, labels_v2d),
            metrics.fowlkes_mallows_score(labels_true, labels_v2d),
            metrics.homogeneity_score(labels_true, labels_v2d),
            metrics.normalized_mutual_info_score(labels_true, labels_v2d),
            metrics.v_measure_score(labels_true, labels_v2d)
      )

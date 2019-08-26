from sklearn import metrics
import os
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile


in_path = os.path.join(os.pardir, "experiment", "clusters.xlsx")
df = pd.read_excel(in_path, sheet_name='memmove&memcpy')

# a = df['our model']
# b = df['ground truth']
# a = list(a)
# print(type(a))
# print(a)

labels_true = df['truth']
labels_pred = df['model']
labels_bert = df['bert-12']


preds = (("model", labels_pred))




print("ARI	COM	FMI	HOM	NMI	V-M")

print(
      metrics.adjusted_rand_score(labels_true, labels_pred),
      metrics.completeness_score(labels_true, labels_pred),
      metrics.fowlkes_mallows_score(labels_true, labels_pred),
      metrics.homogeneity_score(labels_true, labels_pred),
      metrics.normalized_mutual_info_score(labels_true, labels_pred),
      metrics.v_measure_score(labels_true, labels_pred)
)

print(
      metrics.adjusted_rand_score(labels_true, labels_bert),
      metrics.completeness_score(labels_true, labels_bert),
      metrics.fowlkes_mallows_score(labels_true, labels_bert),
      metrics.homogeneity_score(labels_true, labels_bert),
      metrics.normalized_mutual_info_score(labels_true, labels_bert),
      metrics.v_measure_score(labels_true, labels_bert)
)

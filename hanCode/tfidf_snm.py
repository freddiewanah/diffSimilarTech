import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn import svm
from sklearn.metrics import classification_report

# read files
trainData = pd.read_csv("../data/train_data2.csv")
testData = pd.read_csv("../data/test_data3.csv")

# Create feature vectors
vectorizer = TfidfVectorizer(min_df=5,
                             max_df=0.8,
                             sublinear_tf=True,
                             use_idf=True)

train_vectors = vectorizer.fit_transform(trainData['sentence'])
test_vectors = vectorizer.transform(testData['sentence'])

# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(train_vectors, trainData['polarity'])
t1 = time.time()
prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1

# results
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))

report = classification_report(testData['polarity'], prediction_linear, output_dict=True)

print(report)
print('0: ', report['0'])
print('1: ', report['1'])
print('2: ', report['2'])
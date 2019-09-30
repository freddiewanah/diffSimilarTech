from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import pandas as pd
from sklearn.metrics import classification_report

# read files
trainData = pd.read_csv("../data/train_data2.csv")
testData = pd.read_csv("../data/test_data3.csv")


stop_words = ['in', 'of', 'at', 'a', 'the']
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3), stop_words=stop_words)
ngram_vectorizer.fit(trainData['sentence'])
X = ngram_vectorizer.transform(trainData['sentence'])
X_test = ngram_vectorizer.transform(testData['sentence'])

X_train, X_val, y_train, y_val = train_test_split(
    X, trainData['polarity'], train_size=0.75
)

for c in [0.001, 0.005, 0.01, 0.05, 0.1]:
    svm = LinearSVC(C=c)
    svm.fit(X_train, y_train)
    print("Accuracy for C=%s: %s"
          % (c, accuracy_score(y_val, svm.predict(X_val))))


final = LinearSVC(C=0.1)
final.fit(X, trainData['polarity'])
print(final.predict(X_test))
print("Final Accuracy: %s"
      % accuracy_score(testData['polarity'], final.predict(X_test)))
report = classification_report(testData['polarity'], final.predict(X_test), output_dict=True)
print(report)
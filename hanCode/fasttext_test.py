import pandas as pd
import os
import fasttext


# read files
trainData = pd.read_csv("../data/train_data2.csv")
testData = pd.read_csv("../data/test_data3.csv")

file = open(os.path.join(os.pardir, "outnew", "trainFastData.txt"),"w+")
for idx in range(len(trainData['sentence'])):
    line = "_label_"+str(trainData['polarity'][idx])+' '+trainData['sentence'][idx]
    file.write(line+'\n')

model = fasttext.train_supervised(os.path.join(os.pardir, "outnew", "trainFastData.txt"))


c = 0
file1 = open(os.path.join(os.pardir, "outnew", "testFastData.txt"),"w+")
for idx in range(len(testData['sentence'])):
    sen = testData['sentence'][idx]
    line = testData['sentence'][idx]
    file1.write(line+'\n')


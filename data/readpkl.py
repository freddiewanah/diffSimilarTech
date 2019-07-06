import pickle
import os
f = open(os.path.join(os.pardir, "outnew", "sentences.pkl"),'rb')
data = pickle.load(f)
print(data) 

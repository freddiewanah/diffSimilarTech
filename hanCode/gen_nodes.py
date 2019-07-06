import os
import pickle
from nltk.parse import CoreNLPParser
from nltk import pos_tag
from gensim.similarities import WmdSimilarity
import gensim
import datetime

ver_flag= False
print(datetime.datetime.now())
def set_threshold(a, b):
    if ver_flag:
        if a == b:
            return 0.52
        return 0.55 - 0.05 ** abs(a - b)
    else:
        if a == b:
            return 0.55
        elif a > 3 or b > 3:
            return 0.55 - 0.1 ** abs(a - b)
        return 0.55 - 0.05 ** abs(a - b)

# Prepare sentences
def read_relation(path):
    """ Read relation files and process

        (str) -> (dict)
    """

    file_path = os.path.join(os.pardir, "out", path, "relations.pkl")
    relations_file = open(file_path, 'rb')
    relations = pickle.load(relations_file)
    relations_file.close()
    return relations


# Read comparative sentences
stackoverflow = read_relation("stackoverflow_v1")

sentences = []
flag = True  # all sentences
# Prepare stop words set
stop_words = pickle.load(open(os.path.join(os.pardir, "data", "stop_words.pkl"), 'rb'))
# Prepare POS tagger
pos_tag_set = {"JJR", "JJ", "NN", "NNS", "NNP", "NNPS", "RBR", "RBS", "JJS"}
# Prepare stop phrases
stop_phrases = [["for", "example"], ["in", "terms", "of"], ["keep", "in", "mind"],
                ["in", "this", "case"],
                ["a", "bit"], ["of", "course"], ["due", "to"], ["generally", "speaking"],
                ["in", "general"], ["at", "the", "moment"], ["from", "my", "point", "of", "view"],
                ["in", "my", "experience"], ["at", "least"], ["at", "most"],
                ["from", "my", "experience"], ["in", "so", "many", "ways"],
                ["hard", "data"], ["sorted", "data"], ["unsorted", "data"],
                ["by", "index"], ["new", "element"], ["are", "familiar", "of"],
                ["ios", "google-chrome"], ["several", "tests"]]

pos_flag = True

# Process sentences
for pair in stackoverflow:
    cp = [sentence[-1] for sentence in stackoverflow[pair]]
    for s in cp:
        s = s.replace(pair[0], "A")
        s = s.replace(pair[1], "B")
        if s not in sentences:
            sentences.append(s)

l = len(sentences)
corpus = []
topics = []
for sentence in sentences:
    if pos_flag:
        words = sentence.split()
        words[-1] = words[-1].strip()
        tagged_words = CoreNLPParser(url='http://localhost:9000', tagtype='pos').tag(words)
        if len(words) != len(tagged_words):
            tagged_words = pos_tag(words)
        # print(tagged_words)
        # print(sentence.strip())
        for phrase in stop_phrases:
            n = len(phrase)
            for i in range(len(tagged_words) - n + 1):
                if phrase == words[i:i+n]:
                    for j in range(i, i+n):
                        tagged_words[j] = (None, tagged_words[j][1])
        i = 0
        indices = []
        keywords = []
        for (word, tag) in tagged_words:
            if word in pair:
                indices.append(i)
                keywords.append(word)
                i += 1
            elif word not in stop_words and tag in pos_tag_set and word is not None:
                keywords.append(word)
                i += 1
        # topics.append(" ".join(keywords))
        # topics.append(sentence.strip())
        if len(keywords) <= 10 and flag:
            ws = [w for w in keywords if w != 'A' and w != 'B']
        else:
            ws = []
            # if len(indices) == 2:
            #     for j in range(len(keywords)):
            #
            #         if j > indices[0] and j <= indices[0] + 4 and keywords[j] not in pair and j < indices[1]:
            #             ws.append(keywords[j])
            #         elif j >= indices[1] - 2 and j <= indices[1] + 2 and keywords[j] not in pair:
            #             ws.append(keywords[j])
            # else:
            if True:
                for j in range(len(keywords)):
                    for i in indices:
                        if j >= i - 2 and j <= i + 2 and keywords[j] not in pair and keywords[j] not in ws:
                            ws.append(keywords[j])
                            break
        # with open(keywords_path, "a") as keywords_file:
        #     keywords_file.write(",".join(ws)+"\n")
        #     keywords_file.write(sentence+"\n")
        corpus.append(ws)
        topics.append(" ".join(ws))
    else:
        corpus.append([w for w in sentence.split() if w not in stop_words])

# with open(os.path.join(os.pardir, "outnew", "corpus.pkl"), 'wb') as corpus_file:
#     pickle.dump(corpus, corpus_file)
# with open(os.path.join(os.pardir, "outnew", "sentences.pkl"), 'wb') as sentences_file:
#     pickle.dump(sentences, sentences_file)

print(datetime.datetime.now())
print("finished gathering sentences")
# Prepare word2vector model
# model = gensim.models.Word2Vec(sentences, min_count=20, size=200, workers=8)
print(1)
print(2)
fname = os.path.join(os.pardir, "data", "mymodel")
model = gensim.models.Word2Vec.load(fname)
model.init_sims(replace=True)
print(3)

print(datetime.datetime.now())
print("model inited")

# Build weighted graph
# dictionary = Dictionary(corpus)
# bow_corpus = [dictionary.doc2bow(document) for document in corpus]

print(l)
index = WmdSimilarity(corpus, model)

out_file = open(os.path.join(os.pardir, "outnew", "test_gephi.csv"), "a")
out_file.write("target,source\n")
out_file.close()
count = 0
for i in range(l - 1):
    sims = index[corpus[i]]
    # print("query:")
    # print(corpus[i])
    # print(sentences[i])
    # print("sims:")
    for j in range(i + 1, l):
        # print(sims[j])
        # print(corpus[i])
        # print(corpus[j])
        # print(sentences[j])
        # print()

        threshold = set_threshold(len(corpus[i]), len(corpus[j]))
        if sims[j] >= threshold*1.1:
            count += 1
            print("{} edges".format(count))
            print("{} of {} sentences".format(i+1, l))
            out_file = open(os.path.join(os.pardir, "outnew", "test_gephi.csv"),
                            "a")
            out_file.write("{},{}".format(sentences[i][:-1], sentences[j]))
            out_file.close()
print(datetime.datetime.now())
print("All work done :) ")

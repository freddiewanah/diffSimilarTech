# coding=utf-8

import json
from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost', port=9000, quiet=False)
props = {'annotators': 'coref', 'pipelineLanguage': 'en'}
# pre = 'Barack Obama was born in Hawaii, he is the president.'
# words = 'he was elected in 2008. '
# post = 'Blabla get married to him and she is happy.'

pre = 'Mercurial and Bazaar resemble themselves very much on the surface.'
words = 'they both provide basic distributed version control as in offline commit and merging multiple branches are both written in python and are both slower than git.'
post = 'it is the fastest of the three and it is also the most powerful of the three, by quite a margin.'

text = pre + " "+words+ " "+post

text="qlite just picks the first non-null integer mysql requires auto increment postgresql uses sequences etc. it s a mess and that s only among the oss databases. ry getting oracle microsoft and ibm to collectively decide on a tricky bit of functionality"

print(nlp.coref(text))

pre_list=nlp.word_tokenize(pre)
words_list=nlp.word_tokenize(words)
post_list=nlp.word_tokenize(post)

for ps in nlp.coref(text):
    for index in range(1, len(ps)):
        if ps[index][0] == 1:
            for i in range(ps[index][1]-1, ps[index][2]-1):
                pre_list.remove(pre_list[i])
            pre_list.insert(ps[index][1]-1, ps[0][-1])
        elif ps[index][0] == 2:
            for i in range(ps[index][1]-1, ps[index][2]-1):
                words_list.remove(words_list[i])
            words_list.insert(ps[index][1]-1, ps[0][-1])
        elif ps[index][0] == 3:
            for i in range(ps[index][1]-1, ps[index][2]-1):
                post_list.remove(post_list[i])
            post_list.insert(ps[index][1]-1, ps[0][-1])

pre = ' '.join(pre_list)
words = ' '.join(words_list)
post = ' '.join(post_list)

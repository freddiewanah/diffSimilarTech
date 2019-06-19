"""
Build pattern matcher.
"""

import nltk
import os.path
from nltk.parse import CoreNLPParser
import spacy
from spacy.matcher import Matcher


class PatternMatcher:


    def __init__(self):
        self.count = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0,
                      "6": 0, "7": 0, "8": 0, "9": 0, "10": 0}
        self.compa_sent_count = 0

        self.nlp = spacy.load('en')
        self.matcher = Matcher(self.nlp.vocab)
        # self.matcher.add(0,
        #             None,
        #             [{'ORTH': 'JJR'}, {'ORTH': 'CIN'}, {'ORTH': 'TECH'}],
        #             [{'ORTH': 'JJR'}, {}, {'ORTH': 'CIN'}, {'ORTH': 'TECH'}],
        #             [{'ORTH': 'JJR'}, {'ORTH': 'CIN'}, {}, {'ORTH': 'TECH'}],
        #             [{'ORTH': 'JJR'}, {}, {'ORTH': 'CIN'}, {}, {'ORTH': 'TECH'}])
        # self.matcher.add(1,
        #             None,
        #             [{'ORTH': 'RB'}, {'ORTH': 'JJ'}, {'ORTH': 'CIN'}, {'ORTH': 'TECH'}],
        #             [{'ORTH': 'RB'}, {'ORTH': 'JJ'}, {}, {'ORTH': 'CIN'}, {'ORTH': 'TECH'}])
        # self.matcher.add(8,
        #             None,
        #             [{'ORTH': 'RBR'}, {'ORTH': 'JJ'}, {'ORTH': 'CIN'}, {'ORTH': 'TECH'}],
        #             [{'ORTH': 'RBR'}, {'ORTH': 'JJ'}, {}, {'ORTH': 'CIN'}, {'ORTH': 'TECH'}])
        # self.matcher.add(2,
        #             None,
        #             [{'ORTH': 'CV'}, {'ORTH': 'CIN'}, {'ORTH': 'TECH'}],
        #             [{'ORTH': 'CV'}, {}, {'ORTH': 'CIN'}, {'ORTH': 'TECH'}])
        # self.matcher.add(3,
        #             None,
        #             [{'ORTH': 'CV'}, {'ORTH': 'VBG'}, {'ORTH': 'TECH'}])
        # self.matcher.add(4,
        #             None,
        #             [{'ORTH': 'CV'}, {'ORTH': 'TECH'}])
        self.matcher.add(2,
                    None,
                    [{'ORTH': 'VB'}, {'ORTH': 'VBN'}, {'ORTH': 'TECH'}],
                    [{'ORTH': 'VB'}, {'ORTH': 'VBN'}, {}, {'ORTH': 'TECH'}])
        # self.matcher.add(6,
        #             None,
        #             [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {'ORTH': 'JJS'}],
        #             [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {'ORTH': 'JJS'}],
        #             [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'JJS'}],
        #             [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'JJS'}])
        # self.matcher.add(10,
        #             None,
        #             [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {'ORTH': 'RBR'}],
        #             [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {'ORTH': 'RBR'}],
        #             [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'RBR'}],
        #             [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'RBR'}])
        self.matcher.add(0,
                    None,
                    [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {'ORTH': 'JJR'}],
                    [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {'ORTH': 'JJR'}],
                    [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'JJR'}],
                    [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'JJR'}])
        self.matcher.add(1,
                    None,
                    [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {'ORTH': 'JJ'}],
                    [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {'ORTH': 'JJ'}],
                    [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'JJ'}],
                    [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'JJ'}])
        # self.matcher.add(9,
        #             None,
        #             [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {'ORTH': 'RBS'}],
        #             [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {'ORTH': 'RBS'}],
        #             [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'RBS'}],
        #             [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'RBS'}])


    def add_pos_tag(self, words, tech_pair):
        tagged_words = CoreNLPParser(url='http://localhost:9000', tagtype='pos').tag(words)
        # print tagged_words
        tag_list = []
        for (word, tag) in tagged_words:
            if word in tech_pair.split("\t"):
                tag_list.append("TECH")
            else:
                tag_list.append(tag)
        return tag_list

    def match_pattern(self, pre, words, post, current_id, tech_pair):
        tag_list = self.add_pos_tag(words, tech_pair)
        patterns = self.matcher(self.nlp(u'{}'.format(" ".join(tag_list))))
        if patterns != []:
            self.compa_sent_count += 1
            print("yes")
            out_file = open(os.path.join(os.pardir, "outnew", "pattern", "sentences.txt"), "a")
            out_file.write("{}\n".format(current_id))
            out_file.write("{}\n".format(current_id))
            out_file.write("{}\nPattern(s): ".format(tech_pair))
            out_file.write(" ".join(words))
            out_file.write("\n")
            out_file.close()
            data = open(os.path.join(os.pardir, "outnew", "pattern", "output.txt"), "a")
            data.write("{}\n".format(current_id))
            data.write("{}\nPattern(s): ".format(tech_pair))
            for pattern in patterns:
                self.count[str(pattern[0])] += 1
                data.write(str(pattern[0])+"\t")
                # data_file = open(os.path.join(os.pardir, "out", "tech_v2", "{}.txt".format(pattern[0])), "a")
            data.write("\n")
            data.write(" ".join(words))
            data.write("\n\n\n")
            data.close()

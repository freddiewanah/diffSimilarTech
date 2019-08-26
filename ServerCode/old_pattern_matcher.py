"""
Build pattern matcher.
"""

import nltk
import os.path
import spacy
from spacy.matcher import Matcher
import sys


class OldPatternMatcher:

    cv = {"beat", "beats", "prefer", "prefers", "recommend", "recommends",
          "defeat", "defeats", "kill", "kills", "lead", "leads", "obliterate",
          "obliterates", "outclass", "outclasses", "outdo", "outdoes",
          "outperform", "outperforms", "outplay", "outplays", "overtake",
          "overtakes", "smack", "smacks", "subdue", "subdues", "surpass",
          "surpasses", "trump", "trumps", "win", "wins", "blow", "blows",
          "decimate", "decimates", "destroy", "destroys", "buy", "buys",
          "choose", "chooses", "favor", "favors", "grab", "grabs", "pick",
          "picks", "purchase", "purchases", "select", "selects", "race",
          "races", "compete", "competes", "match", "matches", "compare",
          "compares", "lose", "loses", "suck", "sucks"}
    cin = {"than", "over", "beyond", "upon", "as", "against", "out", "behind",
           "under", "between", "after", "unlike", "with", "by", "opposite"}

    def __init__(self):
        self.count = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0,
                      "6": 0, "7": 0, "8": 0, "9": 0, "10": 0}
        self.compa_sent_count = 0

        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)
        self.matcher.add(0,
                    None,
                    [{'ORTH': 'JJR'}, {'ORTH': 'CIN'}, {'ORTH': 'TECH'}],
                    [{'ORTH': 'JJR'}, {}, {'ORTH': 'CIN'}, {'ORTH': 'TECH'}],
                    [{'ORTH': 'JJR'}, {'ORTH': 'CIN'}, {}, {'ORTH': 'TECH'}],
                    [{'ORTH': 'JJR'}, {}, {'ORTH': 'CIN'}, {}, {'ORTH': 'TECH'}])

        self.matcher.add(8,
                    None,
                    [{'ORTH': 'RBR'}, {'ORTH': 'JJ'}, {'ORTH': 'CIN'}, {'ORTH': 'TECH'}],
                    [{'ORTH': 'RBR'}, {'ORTH': 'JJ'}, {}, {'ORTH': 'CIN'}, {'ORTH': 'TECH'}])
        self.matcher.add(2,
                    None,
                    [{'ORTH': 'CV'}, {'ORTH': 'CIN'}, {'ORTH': 'TECH'}],
                    [{'ORTH': 'CV'}, {}, {'ORTH': 'CIN'}, {'ORTH': 'TECH'}])
        self.matcher.add(3,
                    None,
                    [{'ORTH': 'CV'}, {'ORTH': 'VBG'}, {'ORTH': 'TECH'}])


        # self.matcher.add(6,
        #             None,
        #             [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {'ORTH': 'JJS'}],
        #             [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {'ORTH': 'JJS'}],
        #             [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'JJS'}],
        #             [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'JJS'}])
        self.matcher.add(10,
                    None,
                    [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {'ORTH': 'RBR'}],
                    [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {'ORTH': 'RBR'}],
                    [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'RBR'}],
                    [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'RBR'}])
        self.matcher.add(7,
                    None,
                    [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {'ORTH': 'JJR'}],
                    [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {'ORTH': 'JJR'}],
                    [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'JJR'}],
                    [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'JJR'}])
        # self.matcher.add(9,
        #             None,
        #             [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {'ORTH': 'RBS'}],
        #             [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {'ORTH': 'RBS'}],
        #             [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'RBS'}],
        #             [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'RBS'}])


    def add_pos_tag(self, words, table, tech_pair):

        tagged_words = nltk.pos_tag(words.split())
        # print(words)
        # print (tagged_words)
        tag_list = []
        for (word, tag) in tagged_words:
            if tag == "IN" and word in self.cin:
                tag_list.append("CIN")
            elif tag[:2] == "VB" and word in self.cv:
                tag_list.append("CV")
            # else:
            #     self.cursor.execute("SELECT * FROM {} WHERE TagName = \'{}\'".format(table, word))
            #     if self.cursor.rowcount == 0:
            #         tag_list.append(tag)
            #     else:
            #         tag_list.append("TECH")
            elif word == tech_pair.split()[0] or word == tech_pair.split()[1]:
                tag_list.append("TECH")
            else:
                tag_list.append(tag)
        return tag_list

    def old_match_pattern(self, words, current_id, tech_pair, table, words_ori):
        print("check old match pattern", current_id)

        tag_list = self.add_pos_tag(words, table, tech_pair)
        tag_list_ori = self.add_pos_tag(words_ori, table, tech_pair)
        # print(words, tag_list, tech_pair)
        # print(tech_pair.split()[0])
        # print(tech_pair.split()[1])
        # if tech_pair[0] in words_ori and tech_pair[1] in words_ori:
        #
        #     patterns_ori = self.matcher(self.nlp(u'{}'.format(" ".join(tag_list_ori))))
        #     if patterns_ori != []:
        #         return
        patterns = self.matcher(self.nlp(u'{}'.format(" ".join(tag_list))))
        if patterns != []:
            self.compa_sent_count += 1
            out_file = open(os.path.join(os.pardir, "outnew", "tech_v4", "sentences_500000.txt"), "a")
            out_file.write(words)
            out_file.write("\n")
            out_file.close()
            data_file = open(os.path.join(os.pardir, "outnew", "tech_v4", "output_500000_leased_{}.txt".format(os.getpid())), "a")
            data_file.write("{}\n".format(current_id))
            data_file.write("{}\nPattern(s): ".format(tech_pair))
            for pattern in patterns:
                self.count[str(pattern[0])] += 1
                data_file.write(str(pattern[0])+"\t")
                # data_file = open(os.path.join(os.pardir, "out", "tech_v2", "{}.txt".format(pattern[0])), "a")
            data_file.write("\n")
            data_file.write(words)
            data_file.write("\n\n\n")
            data_file.close()

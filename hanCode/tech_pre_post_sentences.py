"""
Extract sentences containing similar tech pairs, and also pre and post sentences.
"""

import datetime
from multiprocessing import Process
import psycopg2
import operator
import os.path
import pickle
from nltk import pos_tag
from prepros import get_words
from nltk.parse import CoreNLPParser
import spacy
from spacy.matcher import Matcher
from multiprocessing.dummy import Pool as ThreadPool

batch = 500000
s = batch * 8 * 7
table_name = "Posts"

remove_word = [" much ", " a ", " an ", " i "]

selected_tech_pairs = ["ubuntu", "debian", "anjuta", "kdevelop", "postgresql", "mysql", "firefox", "safari", "google-chrome", "firefox", "cassini", "iis", "quicksort", "mergesort", "git", "bazaar", "jython", "pypy", "verilog", "vdhl", "awt", "swing", "vmware", "virtualbox", "vim", "emacs"]

similar_techs_file = open(os.path.join(os.pardir, "data", "similar_techs.pkl"), 'rb')
similar_techs = pickle.load(similar_techs_file)
similar_techs_file.close()

synonyms_file = open(os.path.join(os.pardir, "data", "synonyms.pkl"), 'rb')
synonyms = pickle.load(synonyms_file)
synonyms_file.close()

print(datetime.datetime.now())

class PatternMatcher:


    def __init__(self):
        self.count = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0,
                      "6": 0, "7": 0, "8": 0, "9": 0, "10": 0}
        self.compa_sent_count = 0

        self.nlp = spacy.load("en")
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
        # self.matcher.add(2,
        #             None,
        #             [{'ORTH': 'VB'}, {'ORTH': 'VBN'}, {'ORTH': 'TECH'}],
        #             [{'ORTH': 'VB'}, {'ORTH': 'VBN'}, {}, {'ORTH': 'TECH'}])

        self.matcher.add(4,
                         None,
                         [{'ORTH': 'NN'}, {'ORTH': 'IN'}, {'ORTH': 'TECH'}, {'ORTH': 'VBZ'},  {}, {'ORTH': 'RB'}],
                         [{'ORTH': 'NN'}, {'ORTH': 'IN'}, {'ORTH': 'TECH'}, {'ORTH': 'VBZ'}],
                         [{'ORTH': 'NN'}, {'ORTH': 'IN'}, {'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {'ORTH': 'RB'}],
                         [{'ORTH': 'NN'}, {'ORTH': 'IN'}, {}, {'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {'ORTH': 'RB'}],
                         [{'ORTH': 'NN'}, {'ORTH': 'IN'}, {}, {'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'RB'}],


                         )

        self.matcher.add(5,
                         None,

                         [{'ORTH': 'TECH'}, {'ORTH': 'VBP'}, {'ORTH': 'NN'}],
                         [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBP'}, {'ORTH': 'NN'}],
                         [{'ORTH': 'TECH'}, {'ORTH': 'VBP'}, {}, {'ORTH': 'NN'}],
                         [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBP'}, {}, {'ORTH': 'NN'}],

                         [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {'ORTH': 'NN'}],
                         [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {'ORTH': 'NN'}],
                         [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'NN'}],
                         [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'NN'}],
                         )
        self.matcher.add(6,
                         None,
                         [{'ORTH': 'TECH'}, {'ORTH': 'VBP'}, {'ORTH': 'RB'}],
                         [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBP'}, {'ORTH': 'RB'}],
                         [{'ORTH': 'TECH'}, {'ORTH': 'VBP'}, {}, {'ORTH': 'RB'}],
                         [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBP'}, {}, {'ORTH': 'RB'}],

                         [{'ORTH': 'TECH'}, {'ORTH': 'VBP'}, {'ORTH': 'RBR'}],
                         [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBP'}, {'ORTH': 'RBR'}],
                         [{'ORTH': 'TECH'}, {'ORTH': 'VBP'}, {}, {'ORTH': 'RBR'}],
                         [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBP'}, {}, {'ORTH': 'RBR'}],

                         )

        self.matcher.add(7,
                         None,
                         [{'ORTH': 'NN'}, {'ORTH': 'IN'}, {'ORTH': 'TECH'}],
                         [{'ORTH': 'NN'}, {'ORTH': 'IN'}, {}, {'ORTH': 'TECH'}],


                         )
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
                    [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {}, {}, {}, {'ORTH': 'JJR'}],
                    [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'JJR'}])
        self.matcher.add(1,
                    None,
                    [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {'ORTH': 'JJ'}],
                    [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {'ORTH': 'JJ'}],
                    [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'JJ'}],
                    [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'JJ'}],
                    # [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {}, {}, {}, {'ORTH': 'JJ'}],
                    [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'JJ'}])
        self.matcher.add(3,
                    None,
                    [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {'ORTH': 'RB'}],
                    [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {'ORTH': 'RB'}],
                    [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'RB'}],
                    [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'RB'}],
                    # [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {}, {}, {}, {'ORTH': 'RB'}],
                    [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'RB'}])
        # self.matcher.add(9,
        #             None,
        #             [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {'ORTH': 'RBS'}],
        #             [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {'ORTH': 'RBS'}],
        #             [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'RBS'}],
        #             [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'RBS'}])


    def add_pos_tag(self, words, tech_pair):
        if len(words) == 0:
            return []
        words = words.split()
        tagged_words = CoreNLPParser(url='http://localhost:9000', tagtype='pos').tag(words)
        if len(words) != len(tagged_words):
            tagged_words = pos_tag(words)
        tag_list = []
        for (word, tag) in tagged_words:
            if word in tech_pair:
                tag_list.append("TECH")
            else:
                tag_list.append(tag)
        return tag_list

    def match_pattern(self, pre, words, post, current_id, tech_pair):
        pre_rm = pre
        words_rm = words
        post_rm = post
        for w in remove_word:

            pre_rm = pre_rm.replace(w, ' ')
            words_rm = words_rm.replace(w, ' ')
            post_rm = post_rm.replace(w, ' ')

        tag_list = self.add_pos_tag(words_rm, tech_pair)
        pre_tag_list = self.add_pos_tag(pre_rm, tech_pair)
        post_tag_list = self.add_pos_tag(post_rm, tech_pair)
        words_patterns = []
        pre_patterns = []
        post_patterns = []
        if len(tag_list) > 0:
            words_patterns = self.matcher(self.nlp(u'{}'.format(" ".join(tag_list))))
        if len(pre_tag_list) > 0:
            pre_patterns = self.matcher(self.nlp(u'{}'.format(" ".join(pre_tag_list))))
        if len(post_tag_list) > 0:
            post_patterns = self.matcher(self.nlp(u'{}'.format(" ".join(post_tag_list))))
        patterns = pre_patterns + words_patterns + post_patterns
        if patterns != []:
            self.compa_sent_count += 1
            out_file = open(os.path.join(os.pardir, "outnew", "pattern_v4", "sentences_{}.txt".format(os.getpid())), "a")
            out_file.write("{}\n".format(current_id))
            out_file.write("{}\nPattern(s): \n".format(tech_pair))
            out_file.write("{}\n".format(pre))
            out_file.write("{}\n".format(words))
            out_file.write("{}\n".format(post))
            out_file.close()
            data = open(os.path.join(os.pardir, "outnew", "pattern_v4", "output_{}.txt".format(os.getpid())), "a")
            data.write("{}\n".format(current_id))
            data.write("{}\nPattern(s): ".format(tech_pair))
            for pattern in patterns:
                self.count[str(pattern[0])] += 1
                data.write(str(pattern[0])+"\t")
                # data_file = open(os.path.join(os.pardir, "out", "tech_v2", "{}.txt".format(pattern[0])), "a")
            data.write("\n")
            if pre_patterns != [] or tech_pair[0] in pre or tech_pair[1] in pre:
                data.write("{}\n".format(pre))
            if words_patterns != [] or tech_pair[0] in words or tech_pair[1] in words:
                data.write("{}\n".format(words))
            if post_patterns != [] or tech_pair[0] in post or tech_pair[1] in post:
                data.write("{}\n".format(post))
            data.write("\n\n\n")
            data.close()



def contains_tech(synonym, words):
    """ Test if words contains synonym.

        (str, [str]) -> bool
    """
    if "_" in synonym:
        synonym_list = synonym.split("_")
        n = len(synonym_list)
        for i in range(len(words) - n + 1):
            if synonym_list == words[i:i+n]:
                return True
        return False
    else:
        return synonym in words


def replace_synonym(synonym, tech, words):
    """ Replace the synonym in words with tech.

        (str, str, [str]) -> [str]
    """
    rtn = []
    if "_" in synonym:
        synonym_list = synonym.split("_")
        n = len(synonym_list)
        flag = True
        for i in range(len(words)):
            if i <= len(words) - n and synonym_list == words[i:i+n]:
                rtn.append(tech)
                end = i + n - 1
                flag = False
            elif flag:
                rtn.append(words[i])
            elif i == end:
                flag = True
    else:
        for word in words:
            if word == synonym:
                rtn.append(tech)
            else:
                rtn.append(word)
    return rtn


def check_tech_pairs(pre, words, post):
    """ Test if words contain similar tech pairs and replace synonym with tech.

        ([str]) -> (str, str)
    """
    techs_list = []
    count = 0
    tech_pairs = []
    pre_check = False
    post_check = False

    for first, values in similar_techs.items():
        first_temp = []
        for first_synonym in synonyms[first]:
            if contains_tech(first_synonym, words):
                first_temp.append((first_synonym, first, len(first_synonym)))
        if len(first_temp) != 0:
            for second in values:
                second_temp = []
                for second_synonym in synonyms[second]:
                    if contains_tech(second_synonym, words) or contains_tech(second_synonym, pre) or contains_tech(second_synonym, post):
                        second_temp.append((second_synonym, second, len(second_synonym)))
                if len(second_temp) != 0:
                    count += 1
                    tech_pairs.append((first, second))
                    techs_list += first_temp
                    techs_list += second_temp

    # Replace synonyms with techs in descending order of length.
    for (synonym, tech, l) in sorted(techs_list, key=operator.itemgetter(2), reverse=True):
        if synonym != tech:
            words = replace_synonym(synonym, tech, words)
            pre = replace_synonym(synonym, tech, pre)
            post = replace_synonym(synonym, tech, post)

    rtn = []
    for (first, second) in tech_pairs:
        if first not in selected_tech_pairs or second not in selected_tech_pairs:
            continue
        if first in words and second in words:
            rtn.append(first)
            rtn.append(second)

        else:
            if first in words and second in pre:
                rtn.append(first)
                rtn.append(second)
                pre_check = True
            if first in words and second in post:
                rtn.append(first)
                rtn.append(second)
                post_check = True
    if len(rtn) > 0 and not pre_check and not post_check:
        #return (" ".join(words), "\t".join(rtn)) # (sentence, tech pairs)
        return None
    elif len(rtn) > 0 and (pre_check or post_check):
        return (" ".join(pre), " ".join(words)," ".join(post), "\t".join(rtn))
    else:
        return None


def main(start):
    compa_sent_count = 0
    total_sent_count = 0
    post_count = 0
    current_id = 0
    pattern_matcher = PatternMatcher()
    try:
        pre_words = []
        post_words = []
        conn = psycopg2.connect('dbname=stackoverflow port=5433 host=localhost')
        cursor = conn.cursor()
        query = "SELECT Id, Body FROM {} WHERE Score > 0 AND posttypeid != 1 AND Id >= {} AND Id < {}".format(table_name, start, start+batch)
        # query = "SELECT Id, Body FROM Posts WHERE Id = 11969"
        cursor.execute(query)
        for current_id, row in cursor.fetchall():
            post_count += 1
            word_list = get_words(row)
            total_sent_count += len(word_list)

            for idx in range(0, len(word_list), 2):
                pre_words = word_list[idx-1]
                words = word_list[idx]
                if idx != len(word_list)-1:
                    post_words = word_list[idx+1]
                else:
                    post_words = []
                if words == []:
                    continue

                rtn = check_tech_pairs(pre_words, words, post_words)
                if rtn is not None:
                    if len(rtn)==2:
                        compa_sent_count += 1
                        data_file = open(os.path.join(os.pardir, "outnew", "{}_v4".format(table_name), "{}.txt".format(os.getpid())), "a")
                        data_file.write("{}\n".format(current_id))
                        data_file.write("{}\n".format(rtn[1]))
                        data_file.write("{}\n".format(rtn[0]))
                        data_file.write("\n")
                        data_file.close()
                    else:
                        compa_sent_count += 1

                        data_file = open(os.path.join(os.pardir, "outnew", "{}_v4".format(table_name), "{}.txt".format(os.getpid())), "a")
                        data_file.write("{}\n".format(current_id))
                        data_file.write("{}\n".format(rtn[3]))
                        data_file.write("{}\n".format(rtn[0]))
                        data_file.write("{}\n".format(rtn[1]))
                        data_file.write("{}\n".format(rtn[2]))
                        data_file.write("\n")
                        data_file.close()
                        pattern_matcher.match_pattern(rtn[0], rtn[1], rtn[2], current_id, rtn[3])


    finally:
        print("Proc {}: {}/{} from {} to {} ({} posts)".format(os.getpid(), compa_sent_count, total_sent_count, start, current_id, post_count))

# procs = []
# for i in range(8):
#     proc = Process(target=main, args=(datalist[i],))
#     procs.append(proc)
#     proc.start()
#
# for proc in procs:
#     proc.join()


data = [0]
pool = ThreadPool()
pool.map(main, data)
pool.close()
pool.join()

print(datetime.datetime.now())

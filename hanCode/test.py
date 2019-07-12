import datetime
from multiprocessing import Process
import psycopg2
import operator
import os.path
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle
from nltk import pos_tag
from prepros import get_words
from nltk.parse import CoreNLPParser
import spacy
from spacy.matcher import Matcher
from multiprocessing.dummy import Pool as ThreadPool
from big_tag_group import selected_tags

cin = {"than", "over", "beyond", "upon", "as", "against", "out", "behind",
       "under", "between", "after", "unlike", "with", "by", "opposite"}
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


def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return zip(*[iter(iterable)] * n)


def read_relation(path):
    """ Read relation files and process

        (str) -> (dict)
    """

    file_path = os.path.join(os.pardir, "out", path, "relations.pkl")
    relations_file = open(file_path, 'rb')
    relations = pickle.load(relations_file)
    relations_file.close()
    return relations


sid = SentimentIntensityAnalyzer()
batch = 500000
s = batch * 8 * 7
table_name = "Posts"

remove_word = [" much ", " a ", " an ", " i ", "also", "really"]

selected_tech_pairs = ["ubuntu", "debian", "anjuta", "kdevelop", "postgresql", "mysql", "firefox", "safari",
                       "google-chrome", "firefox", "cassini", "iis", "quicksort", "mergesort", "git", "bazaar",
                       "jython", "pypy", "verilog", "vdhl", "awt", "swing", "vmware", "virtualbox", "vim", "emacs"]

stackoverflow_relations = read_relation("stackoverflow_v1")
available_pairs = stackoverflow_relations.keys()

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
        # self.matcher.add(6,
        #             None,
        #             [{'ORTH': 'JJR'}, {'ORTH': 'CIN'}, {'ORTH': 'TECH'}],
        #             [{'ORTH': 'JJR'}, {}, {'ORTH': 'CIN'}, {'ORTH': 'TECH'}],
        #             [{'ORTH': 'JJR'}, {'ORTH': 'CIN'}, {}, {'ORTH': 'TECH'}],
        #             [{'ORTH': 'JJR'}, {}, {'ORTH': 'CIN'}, {}, {'ORTH': 'TECH'}])
        # self.matcher.add(7,
        #             None,
        #             [{'ORTH': 'RB'}, {'ORTH': 'JJ'}, {'ORTH': 'CIN'}, {'ORTH': 'TECH'}],
        #             [{'ORTH': 'RB'}, {'ORTH': 'JJ'}, {}, {'ORTH': 'CIN'}, {'ORTH': 'TECH'}])
        # self.matcher.add(8,
        #             None,
        #             [{'ORTH': 'RBR'}, {'ORTH': 'JJ'}, {'ORTH': 'CIN'}, {'ORTH': 'TECH'}],
        #             [{'ORTH': 'RBR'}, {'ORTH': 'JJ'}, {}, {'ORTH': 'CIN'}, {'ORTH': 'TECH'}])
        #
        #
        # self.matcher.add(4,
        #                  None,
        #                  [{'ORTH': 'NN'}, {'ORTH': 'IN'}, {'ORTH': 'TECH'}, {'ORTH': 'VBZ'},  {}, {'ORTH': 'RB'}],
        #                  [{'ORTH': 'NN'}, {'ORTH': 'IN'}, {'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}],
        #                  [{'ORTH': 'NN'}, {'ORTH': 'IN'}, {'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {'ORTH': 'RB'}],
        #                  [{'ORTH': 'NN'}, {'ORTH': 'IN'}, {}, {'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {'ORTH': 'RB'}],
        #                  [{'ORTH': 'NN'}, {'ORTH': 'IN'}, {}, {'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'RB'}],
        #
        #
        #                  )
        #
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
        self.matcher.add(1,
                    None,
                    [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {'ORTH': 'JJ'}],
                    [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {'ORTH': 'JJ'}],
                    [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'JJ'}],
                    [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'JJ'}],
                    # [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {}, {}, {}, {'ORTH': 'JJ'}],
                    [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'JJ'}],
                         [{'ORTH': 'TECH'}, {'ORTH': 'VBD'}, {'ORTH': 'JJ'}],
                         [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBD'}, {'ORTH': 'JJ'}],
                         [{'ORTH': 'TECH'}, {'ORTH': 'VBD'}, {}, {'ORTH': 'JJ'}],
                         [{'ORTH': 'TECH'}, {'ORTH': 'VBD'}, {}, {'ORTH': 'JJ'}],
                         # [{'ORTH': 'TECH'}, {'ORTH': 'VBD'}, {}, {}, {}, {'ORTH': 'JJ'}],
                         [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBD'}, {}, {'ORTH': 'JJ'}]
                         )
        self.matcher.add(3,
                    None,
                    [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {'ORTH': 'RB'}],
                    [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {'ORTH': 'RB'}],
                    [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'RB'}],
                    [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'RB'}],
                    # [{'ORTH': 'TECH'}, {'ORTH': 'VBZ'}, {}, {}, {}, {'ORTH': 'RB'}],
                    [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBZ'}, {}, {'ORTH': 'RB'}],
                    [{'ORTH': 'TECH'}, {'ORTH': 'VBD'}, {'ORTH': 'RB'}],
                    [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBD'}, {'ORTH': 'RB'}],
                    [{'ORTH': 'TECH'}, {'ORTH': 'VBD'}, {}, {'ORTH': 'RB'}],
                    [{'ORTH': 'TECH'}, {'ORTH': 'VBD'}, {}, {'ORTH': 'RB'}],
                    # [{'ORTH': 'TECH'}, {'ORTH': 'VBD'}, {}, {}, {}, {'ORTH': 'RB'}],
                    [{'ORTH': 'TECH'}, {}, {'ORTH': 'VBD'}, {}, {'ORTH': 'RB'}]
                         )

    def add_pos_tag(self, words, tech_pair):
        if len(words) == 0:
            return []
        words = words.split()
        tagged_words = CoreNLPParser(url='http://localhost:9000', tagtype='pos').tag(words)
        if len(words) != len(tagged_words):
            tagged_words = pos_tag(words)
        tag_list = []
        for (word, tag) in tagged_words:
            if tag == "IN" and word in cin:
                tag_list.append("CIN")
            elif tag[:2] == "VB" and word in cv:
                tag_list.append("CV")
            elif word == tech_pair.split()[0] or word == tech_pair.split()[1]:
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
        if words_patterns != [] or post_patterns != []:
            pre_ss = sid.polarity_scores("{}".format(pre))
            words_ss = sid.polarity_scores("{}".format(words))
            post_ss = sid.polarity_scores("{}".format(post))
            if ('TECH' in pre_tag_list and 'TECH' in tag_list) and ((pre_ss['compound'] >= 0.05 and words_ss['compound'] <= -0.05) or
             (words_ss['compound'] >= 0.05 and pre_ss['compound'] <= -0.05)) and ((tech_pair[0] in pre and tech_pair[1] in words)
             or (tech_pair[0] in words and tech_pair[1] in pre)):
                self.compa_sent_count += 1
                data = open(os.path.join(os.pardir, "outnew", "pattern_v4", "test_output_{}.txt".format(os.getpid())), "a")
                data.write("{}\n".format(current_id))
                data.write("{}\nPattern(s): ".format(tech_pair))
                for pattern in patterns:
                    self.count[str(pattern[0])] += 1
                    data.write(str(pattern[0]) + "\t")
                data.write("\n")
                data.write("{}\n".format(pre))
                data.write("{}\n".format(words))
                data.write("\n\n\n")
                data.close()
            if ('TECH' in post_tag_list and 'TECH' in tag_list) and (
                    (post_ss['compound'] >= 0.05 and words_ss['compound'] <= -0.05) or
                    (words_ss['compound'] >= 0.05 and post_ss['compound'] <= -0.05)) and \
                    ((tech_pair[0] in post and tech_pair[1] in words) or (tech_pair[0] in words and tech_pair[1] in post)):
                self.compa_sent_count += 1
                data = open(os.path.join(os.pardir, "outnew", "pattern_v4", "test_output_{}.txt".format(os.getpid())), "a")
                data.write("{}\n".format(current_id))
                data.write("{}\nPattern(s): ".format(tech_pair))
                for pattern in patterns:
                    self.count[str(pattern[0])] += 1
                    data.write(str(pattern[0]) + "\t")
                data.write("\n")
                data.write("{}\n".format(words))
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
            if synonym_list == words[i:i + n]:
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
            if i <= len(words) - n and synonym_list == words[i:i + n]:
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
                    if contains_tech(second_synonym, words) or contains_tech(second_synonym, pre) or contains_tech(
                            second_synonym, post):
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
        for selected_tag in selected_tags:
            if first in selected_tag and second in selected_tag:

                if "{} or {}".format(first, second) in words or "{} and {}".format(first, second) in words or "{}, {}".format(
                        first, second) in words or "{} or {}".format(second, first) in words or "{} and {}".format(second,
                                                                                                                   first) in words or "{}, {}".format(
                        second, first) in words:
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
        # return (" ".join(words), "\t".join(rtn)) # (sentence, tech pairs)
        return None
    elif len(rtn) > 0 and (pre_check or post_check):
        return (" ".join(pre), " ".join(words), " ".join(post), "\t".join(rtn))
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
        # query = "SELECT Id, Body FROM {} WHERE Score > 0 AND posttypeid != 1 AND Id >= {} AND Id < {}".format(table_name, start, start+batch)
        query = "SELECT Id, Body FROM Posts WHERE Id = 145200 or Id = 6713"
        cursor.execute(query)
        for current_id, row in cursor.fetchall():
            post_count += 1
            word_list = get_words(row)
            total_sent_count += len(word_list)

            for idx in range(0, len(word_list), 2):
                if idx == 0:
                    pre_words = []
                else:
                    pre_words = word_list[idx - 1]
                words = word_list[idx]
                if idx != len(word_list) - 1:
                    post_words = word_list[idx + 1]
                else:
                    post_words = []
                if words == []:
                    continue

                rtn = check_tech_pairs(pre_words, words, post_words)
                if rtn is not None:
                    if len(rtn) == 2:
                        compa_sent_count += 1
                        data_file = open(
                            os.path.join(os.pardir, "outnew", "{}_v4".format(table_name), "{}.txt".format(os.getpid())),
                            "a")
                        data_file.write("{}\n".format(current_id))
                        data_file.write("{}\n".format(rtn[1]))
                        data_file.write("{}\n".format(rtn[0]))
                        data_file.write("\n")
                        data_file.close()
                    else:
                        compa_sent_count += 1
                        data_file = open(
                            os.path.join(os.pardir, "outnew", "{}_v4".format(table_name), "{}.txt".format(os.getpid())),
                            "a")
                        data_file.write("{}\n".format(current_id))
                        data_file.write("{}\n".format(rtn[3]))
                        data_file.write("{}\n".format(rtn[0]))
                        data_file.write("{}\n".format(rtn[1]))
                        data_file.write("{}\n".format(rtn[2]))
                        data_file.write("\n")
                        data_file.close()
                        pairs = rtn[3].split()
                        known_pairs = []
                        for x, y in grouped(pairs, 2):
                            if [x, y] not in known_pairs and [y, x] not in known_pairs:
                                pattern_matcher.match_pattern(rtn[0], rtn[1], rtn[2], current_id, "{} {}".format(x, y))
                            known_pairs.append([x, y])




    finally:
        print("Proc {}: {}/{} from {} to {} ({} posts)".format(os.getpid(), compa_sent_count, total_sent_count, start,
                                                               current_id, post_count))


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






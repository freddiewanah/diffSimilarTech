"""
Extract comparative sentences based on similar techs and classify according to
different patterns.
"""

import datetime
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import os.path
from pattern_matcher import PatternMatcher
import pickle
from prepros import get_words
import psycopg2
import sys
import operator

print (datetime.datetime.now())

# start_time = time.time()
# compa_sent_count = 0
# total_sent_count = 0
# post_count = 0


# start = 0
# start = 0
# end = 800


similar_techs_file = open(os.path.join(os.pardir, "data", "similar_techs.pkl"), 'rb')
similar_techs = pickle.load(similar_techs_file)
similar_techs_file.close()

synonyms_file = open(os.path.join(os.pardir, "data", "synonyms.pkl"), 'rb')
synonyms = pickle.load(synonyms_file)
synonyms_file.close()

print(datetime.datetime.now())


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


def check_tech_pairs(words):
    """ Test if words contain similar tech pairs and replace synonym with tech.

        ([str]) -> (str, str)
    """
    techs_list = []
    count = 0
    tech_pairs = []
    for first, values in similar_techs.items():
        first_temp = []
        for first_synonym in synonyms[first]:
            if contains_tech(first_synonym, words):
                first_temp.append((first_synonym, first, len(first_synonym)))
        if len(first_temp) != 0:
            for second in values:
                second_temp = []
                for second_synonym in synonyms[second]:
                    if contains_tech(second_synonym, words):
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

    rtn = []
    for (first, second) in tech_pairs:
        if first in words and second in words:
            rtn.append(first)
            rtn.append(second)

    if len(rtn) > 0:
        return (" ".join(words), "\t".join(rtn)) # (sentence, tech pairs)
    else:
        return None


def test_parallel(start):
    pairs_file = open(os.path.join(os.pardir, "data", "pairs.pkl"), 'rb')
    pairs = pickle.load(pairs_file)
    pairs_file.close()

    synonyms_file = open(os.path.join(os.pardir, "data", "synonyms.pkl"), 'rb')
    synonyms = pickle.load(synonyms_file)
    synonyms_file.close()
    pattern_matcher = PatternMatcher()
    connection = psycopg2.connect('dbname=stackoverflow port=5432 host=localhost')
    with connection.cursor() as cursor:
        sql = "SELECT Id, Body FROM Posts WHERE Score >= 0 AND Id >= {} AND Id < {}".format(start, start + 11000000)
        # sql = "SELECT Id, Body FROM Posts WHERE Id = 46543327"
        cursor.execute(sql)
        for i in range(cursor.rowcount):
            # post_count += 1
            current_id, row = cursor.fetchone()
            word_list = get_words(row)
            # total_sent_count += len(word_list)

            for words in word_list:
                rtn = check_tech_pairs(words)
                if rtn is not None:
                    words = rtn[0].split(" ")
                    pattern_matcher.match_pattern(words, current_id, rtn[1], "keytechs")
test_parallel(47611725)

# end_time = time.time()
# summary_file = open(os.path.join(os.pardir, "out", "tech_v4", "summary.txt"), "a")
# summary_file.write("Id from {} to {}\n".format(start, current_id))
# summary_file.write("Comparative sentences: {}\n".format(pattern_matcher.compa_sent_count))
# summary_file.write("Sentence number: {}\n".format(total_sent_count))
# summary_file.write("Post number: {}\n".format(num))
# for key, value in pattern_matcher.count.iteritems():
#     summary_file.write("Pattern {}: {} sentences\n".format(key, value))
# summary_file.write("\n")
# summary_file.close()
# pattern_matcher.connection.close()
print(datetime.datetime.now())

"""
Extract sentences containing similar tech pairs, and also pre and post sentences.
"""

import datetime
from multiprocessing import Process
#import psycopg2
import operator
import psycopg2
import os.path
import pickle
from prepros import get_words
import spacy
from multiprocessing.dummy import Pool as ThreadPool
from old_pattern_matcher import OldPatternMatcher
from lease_filter_pronoun import Check_new_pattern

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
# nlp = StanfordCoreNLP('http://localhost', port=9000, quiet=False, memory='6g')
nlp = spacy.load('en')
# Add neural coref to SpaCy's pipe
import neuralcoref
neuralcoref.add_to_pipe(nlp)

def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return zip(*[iter(iterable)]*n)

def read_relation(path):
    """ Read relation files and process

        (str) -> (dict)
    """

    file_path = os.path.join(os.pardir, "out", path, "relations.pkl")
    relations_file = open(file_path, 'rb')
    relations = pickle.load(relations_file)
    relations_file.close()
    return relations


batch = 10000000
s = batch * 8 * 7
table_name = "Posts"

remove_word = [" much ", " a ", " an ", " i ", " also ", " really ", " s "]

selected_tech_pairs = ["ubuntu", "debian", "anjuta", "kdevelop", "postgresql", "mysql", "firefox", "safari", "google-chrome", "firefox", "cassini", "iis", "quicksort", "mergesort", "git", "bazaar", "jython", "pypy", "verilog", "vdhl", "awt", "swing", "vmware", "virtualbox", "vim", "emacs"]

stackoverflow_relations = read_relation("stackoverflow_v1")
available_pairs = stackoverflow_relations.keys()

similar_techs_file = open(os.path.join(os.pardir, "data", "similar_techs.pkl"), 'rb')
similar_techs = pickle.load(similar_techs_file)
similar_techs_file.close()

synonyms_file = open(os.path.join(os.pardir, "data", "synonyms.pkl"), 'rb')
synonyms = pickle.load(synonyms_file)
synonyms_file.close()

def coreference(pre_words, words, post_words):
    changed = ""
    flag = False
    if len(pre_words) > 0:
        pre_words[0] = pre_words[0].capitalize()
    words[0] = words[0].capitalize()
    if len(post_words) > 0:
        post_words[0] = post_words[0].capitalize()
    text = ' '.join(pre_words) + "; " + ' '.join(words) + "; " + ' '.join(post_words)
    # print(text)
    for t in text.split():
        t = t.strip('.')
        t = t.strip(',')
        t = t.strip('; ')
        if t in similar_techs.keys():
            for j in text.split():
                if j in similar_techs[t]:
                    flag = True
                    break
        if flag:
            break
    # print("yes")
    text = text.replace(". ", " ")
    if flag:
        # pre_list = nlp.word_tokenize(' '.join(pre_words))
        # words_list = nlp.word_tokenize(' '.join(words))
        # post_list = nlp.word_tokenize(' '.join(post_words))
        # # doc = nlp(text)
        # for ps in nlp.coref(text):
        #     print(text)
        #     # print(ps)
        #     for tag in full_list:
        #         if tag in ps[0][-1].split():
        #
        #             ps[0] = (ps[0][0], ps[0][1], ps[0][1]+1, tag)
        #     if ps[0][-1] not in full_list:
        #         continue
        #     # print(ps)
        #     for index in range(1, len(ps)):
        #         if len(ps[index][-1].split()) == 1:
        #             if ps[index][0] == 1 and pre_list != []:
        #                 continue
        #                 # for i in range(ps[index][1] - 1, ps[index][2] - 1):
        #                 #     # print(pre_list, i)
        #                 #     if i < len(pre_list):
        #                 #         pre_list.remove(pre_list[i])
        #                 # pre_list.insert(ps[index][1] - 1, ps[0][-1])
        #                 # changed = "pre"
        #             elif ps[index][0] == 2 and pre_list != [] and ps[0][-1].lower() != ps[index][-1].lower():
        #                 for i in range(ps[index][1] - 1, ps[index][2] - 1):
        #                     if i < len(words_list):
        #                         words_list.remove(words_list[i])
        #                 words_list.insert(ps[index][1] - 1, ps[0][-1])
        #                 changed = "words"
        #             elif ps[index][0] == 3 and post_list != [] and ps[0][-1].lower() != ps[index][-1].lower():
        #                 for i in range(ps[index][1] - 1, ps[index][2] - 1):
        #                     # print(post_list, i)
        #                     if i < len(post_list):
        #                         post_list.remove(post_list[i])
        #                 post_list.insert(ps[index][1] - 1, ps[0][-1])
        #                 changed = "post"
        # print(text)
        doc = nlp(text)
        check_key = False
        if doc._.has_coref:
            for i in doc._.coref_clusters:
                if str(i.main) in similar_techs.keys():
                    check_key = True
            # if not check_key:
            #     return pre_words, words, post_words, changed
            # print("yes_coref")
            resolved = doc._.coref_resolved
            resolved_list = resolved.split(";")
            for t in range(len(resolved_list)):
                if resolved_list[t] != []:
                    resolved_list[t] = resolved_list[t][1:]
            if len(pre_words) == 0 and len(resolved_list)==3:
                for i in range(len(words)):
                    if words[i] != resolved_list[1].split(' ')[i]:
                        return [], resolved_list[1].split(' '), resolved_list[2].split(' '), "words"
                return [], resolved_list[1].split(' '), resolved_list[2].split(' '), "post"
            else:
                if len(post_words) == 0:
                    return resolved_list[0].split(' '), resolved_list[1].split(' '), [], "words"
                else:
                    for i in range(len(words)):
                        if words[i] != resolved_list[1].split(' ')[i]:
                            return resolved_list[0].split(' '), resolved_list[1].split(' '), [], "words"
                    return [], resolved_list[1].split(' '), resolved_list[2].split(' '), "post"
        else:
            return pre_words, words, post_words, changed
    else:
        return pre_words, words, post_words, changed


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


def check_tech_pairs(pre, words, post, word_ori, post_ori, current_id):
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
            if contains_tech(first_synonym, words) or contains_tech(first_synonym, pre):
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
            word_ori = replace_synonym(synonym, tech, word_ori)
            post_ori = replace_synonym(synonym, tech, post_ori)
            pre = replace_synonym(synonym, tech, pre)
            post = replace_synonym(synonym, tech, post)

    if len(tech_pairs) ==0:
        return []
    pre, words, post, changed = coreference(pre, words, post)
    # print(pre, words, post, changed)
    Check_new_pattern(pre, words, post, tech_pairs, word_ori, current_id)

    # for i in range(len(word_ori)):
    #     if word_ori[i].lower().strip() != words[i].lower().strip():
    #         word_diff = True
    #         break
    #
    # for i in range(len(post_ori)):
    #     if post_ori[i].lower().strip() != post[i].lower().strip():
    #         post_diff = True
    #         break
    rtn = []
    rtn_post=[]
    pos_check = False
    for (first, second) in tech_pairs:

        if "{} or {}".format(first, second) in words or "{} and {}".format(first, second) in words or "{}, {}".format(first, second) in words or "{} or {}".format(second, first) in words or "{} and {}".format(second, first) in words or "{}, {}".format(second, first) in words:
        	continue
        if first in words and second in words and changed == "words":
            rtn.append(first)
            rtn.append(second)
        if first in post and second in post and changed == "post":
            pos_check = True
            rtn_post.append(first)
            rtn_post.append(second)
                # else:
                #     if first in words and second in pre:
                #         rtn.append(first)
                #         rtn.append(second)
                #         pre_check = True
                #     if first in words and second in post:
                #         rtn.append(first)
                #         rtn.append(second)
                #         post_check = True
    return_list = []
    if len(rtn) > 0 or len(rtn_post) > 0:
        if not pos_check:
            words[0] = words[0].lower()
            return_list.append((" ".join(words), "\t".join(rtn), "word", " ".join(word_ori))) # (sentence, tech pairs)
        if pos_check:
            post[0] = post[0].lower()
            return_list.append((" ".join(post), "\t".join(rtn_post), " post", " ".join(post_ori)))
        # return None
    return return_list


def main(start):
    compa_sent_count = 0
    total_sent_count = 0
    post_count = 0
    current_id = 0
    old_pattern_matcher = OldPatternMatcher()

    try:
        pre_words = []
        post_words = []
        conn = psycopg2.connect('dbname=stackoverflow port=5433 host=localhost')
        cursor = conn.cursor()
        query = "SELECT Id, Body FROM {} WHERE Score = 0 AND posttypeid != 1 AND Id >= {} AND Id < {}".format(table_name, start, start+batch)
        # query = "SELECT Id, Body FROM Posts WHERE Id = 90444"
        # query = "SELECT Id, Body FROM Posts WHERE Id = 115838 "
        cursor.execute(query)

        for current_id, row in cursor.fetchall():

            post_count += 1
            word_list = get_words(row)
            total_sent_count += len(word_list)

            for idx in range(0, len(word_list), 2):
                if idx == 0:
                    pre_words = []
                else:
                    pre_words = word_list[idx-1]
                words = word_list[idx]
                if idx != len(word_list)-1:
                    post_words = word_list[idx+1]
                else:
                    post_words = []
                if words == []:
                    continue

                rtns = check_tech_pairs(pre_words, words, post_words, words, post_words,
                                        current_id)
                for rtn in rtns:
                    if rtn is not None:
                        if len(rtn)==4:
                            compa_sent_count += 1
                            data_file = open(os.path.join(os.pardir, "outnew", "oldPattern_{}_v4".format(table_name), "changed_{}.txt".format(os.getpid())), "a")
                            data_file.write("{}\n".format(current_id))
                            data_file.write("{}\n".format(rtn[1]))
                            data_file.write("Changed: \n{}\n".format(rtn[0]))
                            if rtn[2] == "word":
                                data_file.write("Origin: \n{}\n".format(' '.join(words)))
                            else:
                                data_file.write("Origin: \n{}\n".format(' '.join(post_words)))
                            data_file.write("\n\n")
                            data_file.close()
                            old_pattern_matcher.old_match_pattern(rtn[0], current_id, rtn[1], table_name, rtn[-1])
                        else:
                            compa_sent_count += 1
                            data_file = open(os.path.join(os.pardir, "outnew", "{}_v4".format(table_name), "leased_{}.txt".format(os.getpid())), "a")
                            data_file.write("{}\n".format(current_id))
                            data_file.write("{}\n".format(rtn[3]))
                            data_file.write("{}\n".format(rtn[0]))
                            data_file.write("{}\n".format(rtn[1]))
                            data_file.write("{}\n".format(rtn[2]))
                            data_file.write("\n")
                            data_file.close()





    finally:
        print("Proc {}: {}/{} from {} to {} ({} posts)".format(os.getpid(), compa_sent_count, total_sent_count, start, current_id, post_count))


datalist =[50097354]

# procs = []
# for i in range(3):
#     proc = Process(target=main, args=(datalist[i],))
#     procs.append(proc)
#     proc.start()
#
# for proc in procs:
#     proc.join()



pool = ThreadPool()
pool.map(main, datalist)
pool.close()
pool.join()

print(datetime.datetime.now())

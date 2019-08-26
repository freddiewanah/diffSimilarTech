import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
hotel_rev = ['we didn t want people to feel uncomfortable writing tests while getting to know testng because we wanted them to keep writing a lot of tests',
             'cassini does not support https']

sid = SentimentIntensityAnalyzer()
for sentence in hotel_rev:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    print(ss['compound'])
    for k in ss:
        print('{0}: {1}, '.format(k, ss[k]), end = '')
        print()

"""
Extract sentences containing similar tech pairs, and also pre and post sentences.
"""
from textblob import TextBlob

print(TextBlob(hotel_rev[0]).sentiment)


def lcs(s1, s2):
    s1 = s1.split()
    s2 = s2.split()
    matrix = [["" for x in range(len(s2))] for x in range(len(s1))]
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                if i == 0 or j == 0:
                    matrix[i][j] = s1[i]
                else:
                    matrix[i][j] = matrix[i-1][j-1] + ' ' +s1[i]
            else:
                matrix[i][j] = max(matrix[i-1][j], matrix[i][j-1], key=len)
    cs = matrix[-1][-1][1:]

    return len(cs.split()), cs

print(lcs("phpunit does not include a html gui ", "simpletest also ships with a very simple html gui "))

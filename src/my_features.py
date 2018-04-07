import pandas as pd
import string
from nltk import pos_tag
from sklearn.base import TransformerMixin


# Parts of Speech Tag Count
class PoS_TagFeatures(TransformerMixin):
    def tag_PoS(self, text):
        text_splited = text.split(' ')
        text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
        text_splited = [s for s in text_splited if s]
        pos_list = pos_tag(text_splited)
        noun_count = len([w for w in pos_list if w[1] in ('NN', 'NNP', 'NNPS', 'NNS')])
        adjective_count = len([w for w in pos_list if w[1] in ('JJ', 'JJR', 'JJS')])
        verb_count = len([w for w in pos_list if w[1] in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')])
        return [noun_count, adjective_count, verb_count]

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        print ("Extracting POS features...")
        return [{'nouns': counts[0],
                 'adjectives': counts[1],
                 'verbs': counts[2]}
                for counts in map(self.tag_PoS, posts)]


# Bad Words Occurance Count
class BadWords_Features(TransformerMixin):
    def badWordCount(self, text):
        badwords = pd.read_csv('../data/bad-words.csv', header=None).iloc[:, 0].tolist()
        badCount = sum(text.count(w) for w in badwords)
        return [badCount, badCount / len(text.split()), badCount / len(text)]

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        print ("Extracting badwords features...")
        return [{'badwordcount': badCounts[0],
                 'normByTotalWords': badCounts[1],
                 'normByTotalChars': badCounts[2]}
                for badCounts in map(self.badWordCount, posts)]


class ExtractedFeatures(TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        return [{'length': len(text),
                 'num_sentences': text.count('.')}
                for text in posts]

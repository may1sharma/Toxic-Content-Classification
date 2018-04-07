import pandas as pd
import nltk

class Data():
    def __init__(self):
        nltk.download('averaged_perceptron_tagger')
        print ("Loading data...")
        self.train = pd.read_csv('../data/Better_Train.csv').fillna(' ')
        self.test = pd.read_csv('../data/tiny_test.csv').fillna(' ')
        self.train_text = self.train['comment_text']
        self.test_text = self.test['comment_text']
        self.classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def get_classes():
    classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    return classes
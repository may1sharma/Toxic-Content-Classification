import numpy as np
import pandas as pd
import pickle

from my_features import PoS_TagFeatures, BadWords_Features, Symbol_Features, TextFeatures
from data_process import Data, get_classes

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def feature_extraction(data, flag):
    # Word Vectorizer
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        ngram_range=(1, 1),
        max_features=20000)

    # N-gram Character Vectorizer
    char_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='char',
        ngram_range=(1, 4),
        max_features=30000)

    # Pipelining Parts of Speech Tag Features with DictVectorizer for processing
    posTag_vectorizer = Pipeline([
        ('parts_of_speech', PoS_TagFeatures()),
        ('dictVect', DictVectorizer(sparse=False))
    ])

    # Pipelining Bad Word Features with DictVectorizer for processing
    badWord_vectorizer = Pipeline([
        ('bad_words', BadWords_Features()),
        ('dictVect', DictVectorizer(sparse=False))
    ])

    # Pipelining Symbol based Features with DictVectorizer for processing
    symbol_vectorizer = Pipeline([
        ('symbols', Symbol_Features()),
        ('dictVect', DictVectorizer(sparse=False))
    ])

    # Pipelining Text Features with DictVectorizer for processing
    text_vectorizer = Pipeline([
        ('texts', TextFeatures()),
        ('dictVect', DictVectorizer(sparse=False))
    ])

    print ("Extracting features...")
    combined_features = FeatureUnion(
        [("word", word_vectorizer), ("char", char_vectorizer), ("pos_tags", posTag_vectorizer),
         ("bad_word", badWord_vectorizer), ("symbol", symbol_vectorizer), ("text", text_vectorizer)])

    if(flag == 'train'):
        features = combined_features.fit(data.train_text).transform(data.train_text)
        print ("Saving features")
        feature_pkl_filename = '../model/features.pkl'
        feature_pkl = open(feature_pkl_filename, 'wb')
        pickle.dump(combined_features, feature_pkl)
        feature_pkl.close()
        print ("Features saved")

    if (flag == 'test'):
        print ("Loading features")
        feature_pkl = open('../model/features.pkl', 'rb')
        loaded_features = pickle.load(feature_pkl)
        print ("Loaded features :: ", loaded_features)
        features = loaded_features.transform(data.test_text)
    return features

def create_and_save():
    data = Data()
    train_features = feature_extraction(data, "train")
    scores = []
    for i in range(len(data.classes)):
        print ("Processing "+data.classes[i])
        train_target = data.train[data.classes[i]]
        classifier = LogisticRegression(solver='sag')
        cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
        scores.append(cv_score)
        print('CV score for class {} is {}'.format(data.classes[i], cv_score))

        print ("Creating model for class "+data.classes[i])
        classifier.fit(train_features, train_target)

        print ("Saving model logistic_regression_%s" %data.classes[i])
        lr_pkl_filename = '../model/logistic_regression_%s.pkl' %data.classes[i]
        lr_model_pkl = open(lr_pkl_filename, 'wb')
        pickle.dump(classifier, lr_model_pkl)
        lr_model_pkl.close()
        print ("Model saved")
    print('Total CV score is {}'.format(np.mean(scores)))
    print ("Successfully created and saved all models!")


def predict_score():
    data = Data()
    test_features = feature_extraction(data, "test")
    submission = pd.DataFrame.from_dict({'id': data.test['id']})
    for i in range(len(data.classes)):
        print ("Processing "+data.classes[i])
        lr_model_pkl = open('../model/logistic_regression_%s.pkl' %data.classes[i], 'rb')
        lr_model = pickle.load(lr_model_pkl)
        print ("Loaded Logistic Regression Model for class %s :: " %data.classes[i], lr_model)
        submission[data.classes[i]] = lr_model.predict_proba(test_features)[:, 1]
    print (submission.head(5))
    print ("Saving output")
    submission.to_csv('../data/output.csv', index=False)
    print ("Output saved")

def predict_individual_score(comment):
    print ("Loading features")
    feature_pkl = open('../model/features.pkl', 'rb')
    loaded_features = pickle.load(feature_pkl)
    print ("Loaded features :: ", loaded_features)
    comment_list = []
    comment_list.append(comment)
    comment_features = loaded_features.transform(comment_list)
    prediction = pd.DataFrame()
    classes = get_classes()
    for i in range(len(classes)):
        print ("Processing "+classes[i])
        lr_model_pkl = open('../model/logistic_regression_%s.pkl' %classes[i], 'rb')
        lr_model = pickle.load(lr_model_pkl)
        print ("Loaded Logistic Regression Model for class %s :: " %classes[i], lr_model)
        prediction[classes[i]] = lr_model.predict_proba(comment_features)[:, 1]
    # print ("Prediction:")
    # print (prediction)
    return prediction
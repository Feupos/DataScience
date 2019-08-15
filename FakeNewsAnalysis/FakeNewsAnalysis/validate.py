import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import matplotlib.dates as md

import sklearn
from sklearn import svm
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from textblob import TextBlob, Word, Blobber
import nltk
import textstat

from lexicalrichness import LexicalRichness

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import random

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import seaborn as sns
sns.set()

from itertools import groupby

import time
from datetime import datetime

import sys
import csv
import ctypes
csv.field_size_limit(int(ctypes.c_ulong(-1).value // 2))

from statistics import mean 

import itertools
def plot_features(body_features, title_features):
    # ['type', 'WC', 'TTR', 'NN', 'quote']
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_dir = 'FakeNewsAnalysis/Results/'
    plt.figure()
    sns.boxplot(y = 'type', x = 'NN', data = body_features, whis="range", palette="vlag")
    plt.tight_layout()
    plt.savefig(save_dir + timestamp + 'body' + 'NN')
    plt.figure()
    sns.boxplot(y = 'type', x = 'TTR', data = body_features, whis="range", palette="vlag")
    plt.tight_layout()
    plt.savefig(save_dir + timestamp + 'body' + 'TTR')
    plt.figure()
    sns.boxplot(y = 'type', x = 'WC', data = body_features, whis="range", palette="vlag")
    plt.tight_layout()
    plt.savefig(save_dir + timestamp + 'body' + 'WC')
    plt.figure()
    sns.boxplot(y = 'type', x = 'quote', data = body_features, whis="range", palette="vlag")
    plt.tight_layout()
    plt.savefig(save_dir + timestamp + 'body' + 'quote')

    plt.figure()
    sns.boxplot(y = 'type', x = 'per_stop', data = title_features, whis="range", palette="vlag")
    plt.tight_layout()
    plt.savefig(save_dir + timestamp + 'title' + 'per_stop')
    plt.figure()
    sns.boxplot(y = 'type', x = 'NN', data = title_features, whis="range", palette="vlag")
    plt.tight_layout()
    plt.savefig(save_dir + timestamp + 'title' + 'NN')
    plt.figure()
    sns.boxplot(y = 'type', x = 'avg_wlen', data = title_features, whis="range", palette="vlag")
    plt.tight_layout()
    plt.savefig(save_dir + timestamp + 'title' + 'avg_wlen')
    plt.figure()
    sns.boxplot(y = 'type', x = 'FK', data = title_features, whis="range", palette="vlag")
    plt.tight_layout()
    plt.savefig(save_dir + timestamp + 'title' + 'FK')

    plt.show()

if __name__ == "__main__":
    title_features = pd.read_pickle('FakeNewsAnalysis/Data/title_features.pkl')
    print(title_features.groupby(['type']).agg(['count']))
    body_features = pd.read_pickle('FakeNewsAnalysis/Data/body_features.pkl') 
    title_features_fake = title_features[title_features.type.isin(['fake'])]
    title_features_reliable = title_features[title_features.type.isin(['reliable'])]
    title_features_fake = title_features_fake[:min(title_features_fake.type.count(),title_features_reliable.type.count())]
    title_features_reliable = title_features_reliable[:min(title_features_fake.type.count(),title_features_reliable.type.count())]
    title_features = title_features_fake.append(title_features_reliable)
    title_features = title_features.sample(frac=1).reset_index(drop=True)
    body_features_fake = body_features[body_features.type.isin(['fake'])]
    body_features_reliable = body_features[body_features.type.isin(['reliable'])]
    body_features_fake = body_features_fake[:min(body_features_fake.type.count(),body_features_reliable.type.count())]
    body_features_reliable = body_features_reliable[:min(body_features_fake.type.count(),body_features_reliable.type.count())]
    body_features = body_features_fake.append(body_features_reliable)
    body_features = body_features.sample(frac=1).reset_index(drop=True)
    #plot_features(body_features, title_features)
    #'per_stop','WC','TTR','NN','avg_wlen','quote','FK','polarity','NNP'
    ignore_features = ['per_stop','avg_wlen','FK','polarity','NNP']
    body_features = body_features.drop(columns=ignore_features)
    title_features = title_features.drop(columns=ignore_features)

    print('number of fake: ' + str(title_features[title_features.type.isin(['fake'])].type.count()) + ' - number of reliable: ' + str(title_features[title_features.type.isin(['reliable'])].type.count()))
    print('baseline: ' + str(title_features[title_features.type.isin(['fake'])].type.count()/title_features.type.count()))

    scores = []
    folds = 5
    cv = sklearn.model_selection.KFold(n_splits=folds)#, random_state=42, shuffle=True)
    for train_index, test_index in cv.split(body_features):
        X = body_features.iloc[train_index].drop(columns=['type'])
        y = body_features.iloc[train_index].type
        clf = svm.SVC(gamma='auto',kernel='linear')
        #clf = GaussianNB()
        clf.fit(X, y) 
        prediction = clf.predict(body_features.iloc[test_index].drop(columns=['type']))
        correct = 0
        total = 0
        for predicted, real in zip(prediction, body_features.iloc[train_index].type):
            total = total + 1
            if (predicted == real):
                correct = correct + 1
        #print('body guess ' + str(correct) + ' out of ' + str(total) + ' - ' + str(correct/total) +'%')
        scores.append(correct/total)
    print(scores)
    print('body feature prediction score: ' + str(mean(scores)))

    scores = []
    cv = sklearn.model_selection.KFold(n_splits=folds)#, random_state=42, shuffle=True)
    for train_index, test_index in cv.split(title_features):
        X = title_features.iloc[train_index].drop(columns=['type'])
        y = title_features.iloc[train_index].type
        clf = svm.SVC(gamma='auto',kernel='linear')
        #clf = GaussianNB()
        clf.fit(X, y) 
        prediction = clf.predict(title_features.iloc[test_index].drop(columns=['type']))
        correct = 0
        total = 0
        for predicted, real in zip(prediction, title_features.iloc[train_index].type):
            total = total + 1
            if (predicted == real):
                correct = correct + 1
        #print('title guess ' + str(correct) + ' out of ' + str(total) + ' - ' + str(correct/total) +'%')
        scores.append(correct/total)
    print(scores)
    print('title feature prediction score: ' + str(mean(scores)))

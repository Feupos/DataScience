import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import matplotlib.dates as md

import sklearn
from sklearn import svm
from sklearn.naive_bayes import GaussianNB, ComplementNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

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

from tqdm import tqdm

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
    sns.boxplot(y = 'type', x = 'per_stop', data = body_features, whis="range", palette="vlag")
    plt.tight_layout()
    plt.savefig(save_dir + timestamp + 'body' + 'per_stop')
    plt.figure()
    sns.boxplot(y = 'type', x = 'NN', data = body_features, whis="range", palette="vlag")
    plt.tight_layout()
    plt.savefig(save_dir + timestamp + 'body' + 'NN')
    plt.figure()
    sns.boxplot(y = 'type', x = 'avg_wlen', data = body_features, whis="range", palette="vlag")
    plt.tight_layout()
    plt.savefig(save_dir + timestamp + 'body' + 'avg_wlen')
    plt.figure()
    sns.boxplot(y = 'type', x = 'FK', data = body_features, whis="range", palette="vlag")
    plt.tight_layout()
    plt.savefig(save_dir + timestamp + 'body' + 'FK')

    plt.figure()
    sns.boxplot(y = 'type', x = 'NN', data = title_features, whis="range", palette="vlag")
    plt.tight_layout()
    plt.savefig(save_dir + timestamp + 'title' + 'NN')
    plt.figure()
    sns.boxplot(y = 'type', x = 'TTR', data = title_features, whis="range", palette="vlag")
    plt.tight_layout()
    plt.savefig(save_dir + timestamp + 'title' + 'TTR')
    plt.figure()
    sns.boxplot(y = 'type', x = 'WC', data = title_features, whis="range", palette="vlag")
    plt.tight_layout()
    plt.savefig(save_dir + timestamp + 'title' + 'WC')
    plt.figure()
    sns.boxplot(y = 'type', x = 'quote', data = title_features, whis="range", palette="vlag")
    plt.tight_layout()
    plt.savefig(save_dir + timestamp + 'title' + 'quote')
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
    group_a_type = 'fake'
    group_b_type = 'reliable'
    title_features = pd.read_pickle('FakeNewsAnalysis/Data/title_features.pkl')
    #print(title_features.groupby(['type']).agg(['count']))
    body_features = pd.read_pickle('FakeNewsAnalysis/Data/body_features.pkl') 
    title_features_fake = title_features[title_features.type.isin([group_a_type])]
    title_features_reliable = title_features[title_features.type.isin([group_b_type])]
    title_features_fake = title_features_fake[:750]#:min(title_features_fake.type.count(),title_features_reliable.type.count())]
    title_features_reliable = title_features_reliable[:750]#:min(title_features_fake.type.count(),title_features_reliable.type.count())]
    title_features = title_features_fake.append(title_features_reliable)
    title_features = title_features.sample(frac=1).reset_index(drop=True)
    body_features_fake = body_features[body_features.type.isin([group_a_type])]
    body_features_reliable = body_features[body_features.type.isin([group_b_type])]
    body_features_fake = body_features_fake[:750]#:min(body_features_fake.type.count(),body_features_reliable.type.count())]
    body_features_reliable = body_features_reliable[:750]#:min(body_features_fake.type.count(),body_features_reliable.type.count())]
    body_features = body_features_fake.append(body_features_reliable)
    body_features = body_features.sample(frac=1).reset_index(drop=True)
    plot_features(body_features, title_features)
    #'per_stop','WC','TTR','NN','avg_wlen','quote','FK','polarity','NNP'
    ignore_features = ['polarity', 'NNP']#'per_stop','avg_wlen','FK','polarity','NNP']
    full_body_features =  body_features#.drop(columns=ignore_features)
    full_title_features =  title_features#.drop(columns=ignore_features)

    print('number of ' + group_a_type + ': ' + str(title_features[title_features.type.isin([group_a_type])].type.count()) + ' - number of '+ group_b_type + ': ' + str(title_features[title_features.type.isin([group_b_type])].type.count()))
    print('baseline: ' + str(title_features[title_features.type.isin([group_a_type])].type.count()/title_features.type.count()))
    #features = ['type','per_stop','WC','TTR','NN','avg_wlen','quote','FK','polarity','NNP','str_neg','str_pos','JJR','JJS','RBR','RBS']
    features = ['type','per_stop','WC','TTR','NN','avg_wlen','quote','FK','NNP','str_neg','str_pos','JJR','JJS','RBR','RBS']
    scores = []
    folds = int(title_features[title_features.type.isin([group_a_type])].type.count()/15)
    cv = sklearn.model_selection.KFold(n_splits=folds, shuffle=True)
    body_features = full_body_features[features]#[['type', 'NN', 'TTR', 'WC', 'quote']]
    #features = ['per_stop','WC','TTR','NN','avg_wlen','quote','FK','polarity','NNP','str_neg','str_pos','JJR','JJS','RBR','RBS']

    #print(body_features)
    with tqdm(total=folds) as pbar:
        for train_index, test_index in cv.split(body_features):
            X = body_features.iloc[train_index].drop(columns=['type'])
            y = body_features.iloc[train_index].type
            #clf = svm.SVC(gamma='auto',kernel ='linear')
            #clf = GaussianNB()
            #clf = MultinomialNB()
            clf = RandomForestClassifier(n_estimators=32, max_depth=10)#, min_samples_split = 0.4, min_samples_leaf = 0.05, random_state=42)
            #clf = GradientBoostingClassifier()
            clf.fit(X, y) 
            scores.append(clf.score(body_features.iloc[test_index].drop(columns=['type']), body_features.iloc[test_index].type))
            pbar.update(1)
    #print(scores)
    print('body feature prediction score: ' + str(mean(scores)))

    scores = []
    #folds = 5
    cv = sklearn.model_selection.KFold(n_splits=folds, shuffle=True)
    title_features = full_title_features[features]#[['type', 'per_stop', 'NN', 'avg_wlen', 'FK']]
    #print(title_features)
    with tqdm(total=folds) as pbar:
        for train_index, test_index in cv.split(title_features):
            X = title_features.iloc[train_index].drop(columns=['type'])
            y = title_features.iloc[train_index].type
            #clf = svm.SVC(gamma='auto',kernel ='linear')
            #clf = GaussianNB()
            #clf = MultinomialNB()
            clf = RandomForestClassifier(n_estimators=32, max_depth=10)#, min_samples_split = 0.4, min_samples_leaf = 0.05, random_state=42)
            #clf = GradientBoostingClassifier()
            clf.fit(X, y) 
            scores.append(clf.score(title_features.iloc[test_index].drop(columns=['type']), title_features.iloc[test_index].type))
            pbar.update(1)
    #print(scores)
    print('title feature prediction score: ' + str(mean(scores)))


    scores = []
    #folds = 5
    cv = sklearn.model_selection.KFold(n_splits=folds, shuffle=True)
    joint_features = title_features.drop(columns=['type']).add_suffix('_title').join(body_features)
    #print(joint_features)
    with tqdm(total=folds) as pbar:
        for train_index, test_index in cv.split(joint_features):
            X = joint_features.iloc[train_index].drop(columns=['type'])
            y = joint_features.iloc[train_index].type
            #clf = svm.SVC(gamma='auto',kernel ='linear')
            #clf = GaussianNB()
            #clf = MultinomialNB()
            clf = RandomForestClassifier(n_estimators=32, max_depth=10)#, min_samples_split = 0.4, min_samples_leaf = 0.05, random_state=42)
            #clf = GradientBoostingClassifier()
            clf.fit(X, y) 
            scores.append(clf.score(joint_features.iloc[test_index].drop(columns=['type']), joint_features.iloc[test_index].type))
            pbar.update(1)
    #print(scores)
    print('joint feature prediction score: ' + str(mean(scores)))

# avaliar performance com as mesmas features
## testar com o mesmo datase!!!!!
# especular motivos para diferenca
# adicionar features do LIWC
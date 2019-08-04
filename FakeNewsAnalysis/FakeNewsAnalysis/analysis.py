import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import matplotlib.dates as md

import sklearn
from sklearn import svm

from textblob import TextBlob, Word, Blobber
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
import textstat

from lexicalrichness import LexicalRichness

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import random

import plotly
plotly.tools.set_credentials_file(username='feupos', api_key='zplCPm3MX3jp55lj7sMz')
#Plotly Tools
from plotly.offline import init_notebook_mode, iplot
#init_notebook_mode(connected=False)

import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

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
#from multiprocessing.pool import ThreadPool
#pool = ThreadPool(20)  # However many you wish to run in parallel

# initialize data structures
start = time.time()
iteration_count = 0
    

def preprocess_data(data):

    data_out = pd.DataFrame()
    data_out = data[['type','content']]
    data_out.dropna(inplace=True)
    return data_out

def count_words(text):
    try:
        return len(TextBlob(text).words)
    except:
        return 0

def calc_ttr(text):
    try:
        return LexicalRichness(text).ttr
    except:
        return 0

def count_nouns(text):
    try:
        tags = TextBlob(text).tags
        tags = [i[1] for i in tags]
        return sum(map(lambda x : 1 if 'NN' in x else 0, tags))/len(tags)
    except:
        return 0

def count_quotes(text):
    try:
        return sum(map(lambda x : 1 if '"' in x else 0, text))/len(text)
    except:
        return 0

def count_stop(text):
    try:
        words = TextBlob(text).words
        return sum(map(lambda x : 1 if x in stop else 0, words))/len(words)
    except:
        return 0

def avg_wlen(text):
    try:
        words = TextBlob(text).words
        return sum(map(lambda x : len(x), words))/len(words)
    except:
        return 0

def fk_grade(text):
    try:
        return textstat.flesch_kincaid_grade(text)
    except:
        return 0
    #blob = TextBlob(text)
    #return 0.39 * (len(blob.words)/len(blob.sentences)) + 11.8 ((len(blob.sy))/len(blob.words)) -15.59


def parse_features(data, title_features, body_features):
    new_body_features = pd.DataFrame({'type':data['type'],
                                      'WC':data['content'].map(count_words),
                                      'TTR':data['content'].map(calc_ttr),
                                      'NN':data['content'].map(count_nouns),
                                      'quote':data['content'].map(count_quotes)})
    #print(new_body_features)
    body_features = body_features.append(new_body_features)
    #need this for some reason
    body_features['WC'] = body_features['WC'].astype(int)
    new_title_features = pd.DataFrame({'type':data['type'],
                                      'per_stop':data['title'].map(count_stop),
                                      'NN':data['title'].map(count_nouns),
                                      'avg_wlen':data['title'].map(avg_wlen),
                                      'FK':data['title'].map(fk_grade)})
    title_features = title_features.append(new_title_features)

    return title_features, body_features
    #body_features = body_features.reset_index(drop=True)
    #body_features['type'] = body_features['type'].append(data['type'])
    

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

    print("start")

    filename = 'FakeNewsAnalysis/Data/news_full.csv'
    #filename = 'FakeNewsAnalysis/Data/news_sample.csv'

    sample_size = 0.1  # up to 1

    n_rows = 1000
    chunk_size = 10000

    df_chunk = pd.read_csv( filename, chunksize = chunk_size, header = 0, nrows = n_rows,
                            engine='python', skip_blank_lines=True,  error_bad_lines = False)
                            #skiprows=lambda i: i>0 and random.random() > sample_size)

    dataset = pd.DataFrame()
    polarity = pd.DataFrame()
    words_freq_fake = pd.DataFrame()
    words_freq_reliable = pd.DataFrame()

    title_features = pd.DataFrame(columns = ['per_stop', 'NN', 'avg_wlen', 'FK'])
    body_features = pd.DataFrame(columns = ['type', 'WC', 'TTR', 'NN', 'quote'])

    for chunk in df_chunk:
        iteration_count = iteration_count+1
        print('Running iteration: ', iteration_count, "out of: ", int(np.ceil(n_rows/chunk_size)))
        try:
            chunk = chunk[chunk.type.isin(['fake', 'reliable'])]
            chunk = chunk.sample(frac=sample_size)
            title_features, body_features = parse_features(chunk, title_features, body_features)
        except:
            print('Failure in iteration')

    print(body_features)
    print(title_features)
    #plot_features(body_features, title_features)
    end = time.time()
    print("total time: ", end - start)
    print('baseline: ' + str(title_features[title_features.type.isin(['fake'])].type.count()/title_features.type.count()))

    scores = []
    cv = sklearn.model_selection.KFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in cv.split(body_features):
        X = body_features.iloc[train_index].drop(columns='type')
        y = body_features.iloc[train_index].type
        clf = svm.SVC(gamma='scale')
        clf.fit(X, y) 
        prediction = clf.predict(body_features.iloc[test_index].drop(columns='type'))
        correct = 0
        total = 0
        for predicted, real in zip(prediction, body_features.iloc[train_index].type):
            total = total + 1
            if (predicted == real):
                correct = correct + 1
        #print('body guess ' + str(correct) + ' out of ' + str(total) + ' - ' + str(correct/total) +'%')
        scores.append(correct/total)
    print('body feature prediction score: ' + str(mean(scores)))

    scores = []
    cv = sklearn.model_selection.KFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in cv.split(title_features):
        X = title_features.iloc[train_index].drop(columns='type')
        y = title_features.iloc[train_index].type
        clf = svm.SVC(gamma='scale')
        clf.fit(X, y) 
        prediction = clf.predict(title_features.iloc[test_index].drop(columns='type'))
        correct = 0
        total = 0
        for predicted, real in zip(prediction, title_features.iloc[train_index].type):
            total = total + 1
            if (predicted == real):
                correct = correct + 1
        #print('title guess ' + str(correct) + ' out of ' + str(total) + ' - ' + str(correct/total) +'%')
        scores.append(correct/total)
    print('title feature prediction score: ' + str(mean(scores)))

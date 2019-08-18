import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import matplotlib.dates as md

import sklearn
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

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

from tqdm import tqdm

import glob

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
        return sum(map(lambda x : 1 if 'NN' in x else 0, tags))
    except:
        return 0

def count_proper_nouns(text):
    try:
        tags = TextBlob(text).tags
        tags = [i[1] for i in tags]
        return sum(map(lambda x : 1 if 'NNP' in x else 0, tags))
    except:
        return 0

def count_quotes(text):
    try:
        return sum(map(lambda x : 1 if '"' in x else 0, text))/2
    except:
        return 0

def count_per_stop(text):
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

def get_bow(text):
    tfidf_transformer = TfidfTransformer()
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform([text])
    return tfidf_transformer.fit_transform(X_train_counts)
    #bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
    #return bow.fit_transform([text])

def get_polarity(text):
    polarity = 0    
    try:
        polarity = TextBlob(text).sentiment.polarity
    except:
        #print("invalid content for polarity")
        pass
    return polarity


def parse_features(data, title_features, body_features):
    new_body_features = pd.DataFrame({'type':data['type'],
                                      #'BoW':data['content'].map(get_bow),
                                      'per_stop':data['content'].map(count_per_stop),
                                      'WC':data['content'].map(count_words),
                                      'TTR':data['content'].map(calc_ttr),
                                      'NN':data['content'].map(count_nouns),
                                      'avg_wlen':data['content'].map(avg_wlen),
                                      'quote':data['content'].map(count_quotes),
                                      'FK':data['content'].map(fk_grade),
                                      'polarity':data['content'].map(get_polarity),
                                      'NNP':data['content'].map(count_proper_nouns)})
    #print(new_body_features)
    body_features = body_features.append(new_body_features)
    #need this for some reason
    body_features['WC'] = body_features['WC'].astype(int)
    new_title_features = pd.DataFrame({'type':data['type'],
                                      #'BoW':data['title'].map(get_bow),
                                      'per_stop':data['title'].map(count_per_stop),
                                      'WC':data['title'].map(count_words),
                                      'TTR':data['title'].map(calc_ttr),
                                      'NN':data['title'].map(count_nouns),
                                      'avg_wlen':data['title'].map(avg_wlen),
                                      'quote':data['title'].map(count_quotes),
                                      'FK':data['title'].map(fk_grade),
                                      'polarity':data['title'].map(get_polarity),
                                      'NNP':data['title'].map(count_proper_nouns)})
    title_features = title_features.append(new_title_features)

    return title_features, body_features
    #body_features = body_features.reset_index(drop=True)
    #body_features['type'] = body_features['type'].append(data['type'])
    

def parse_full_dataset():
    start = time.time()
    #iteration_count = 0

    print("start")

    filename = 'FakeNewsAnalysis/Data/news_full.csv'
    #filename = 'FakeNewsAnalysis/Data/news_sample.csv'

    sample_size = 0.001  # up to 1

    n_rows = 1000000#84999000000
    chunk_size = 10000

    df_chunk = pd.read_csv( filename, chunksize = chunk_size, header = 0, #nrows = n_rows,
                            engine='python', skip_blank_lines=True,  error_bad_lines = False)
                            #skiprows=lambda i: i>0 and random.random() > sample_size)

    #dataset = pd.DataFrame()
    #polarity = pd.DataFrame()
    #words_freq_fake = pd.DataFrame()
    #words_freq_reliable = pd.DataFrame()

    title_features = pd.DataFrame()#columns = ['type', 'BoW', 'per_stop', 'NN', 'avg_wlen', 'FK'])
    body_features = pd.DataFrame()#columns = ['type', 'BoW', 'WC', 'TTR', 'NN', 'quote'])

    with tqdm(total=n_rows) as pbar:
        for chunk in df_chunk:
            #iteration_count = iteration_count+1
            #print('Running iteration: ', iteration_count, "out of: ", int(np.ceil(n_rows/chunk_size)))
            try:
                #chunk = chunk[chunk.type.isin(['fake', 'reliable'])]
                chunk = chunk.sample(frac=sample_size)
                rows, cols = chunk.shape
                #print('valid entries in iteration: ' + str(rows) + ' out of ' + str(chunk_size*sample_size))
                if (0 < rows):
                    title_features, body_features = parse_features(chunk, title_features, body_features)
            except Exception as error:
                #print('Failure in iteration: ' + str(error))
                pass
            pbar.update(chunk_size)
    pbar.update(chunk_size)
    end = time.time()
    print("total time: ", end - start)

    title_features.to_pickle('FakeNewsAnalysis/Data/title_features.pkl')
    body_features.to_pickle('FakeNewsAnalysis/Data/body_features.pkl') 

def parse_orig_dataset():
    
    title_features = pd.DataFrame()#columns = ['type', 'BoW', 'per_stop', 'NN', 'avg_wlen', 'FK'])
    body_features = pd.DataFrame()#columns = ['type', 'BoW', 'WC', 'TTR', 'NN', 'quote'])

    path_content = r'FakeNewsAnalysis/Data/orig_data/dataset/Fake'
    all_content = glob.glob(path_content + "/*.txt")
    path_title = r'FakeNewsAnalysis/Data/orig_data/dataset/Fake_titles'
    all_title = glob.glob(path_title + "/*.txt")


    df = pd.DataFrame(columns = ['type' , 'content', 'title'])

    for (fc, ft) in zip(all_content, all_title):
        c = open(fc)
        t = open(ft)
        item = pd.DataFrame({'type':['fake'],
                             'title':[t.read()],
                             'content':[c.read()]})
                                    
        df = df.append(item)

    path_content = r'FakeNewsAnalysis/Data/orig_data/dataset/Real'
    all_content = glob.glob(path_content + "/*.txt")
    path_title = r'FakeNewsAnalysis/Data/orig_data/dataset/Real_titles'
    all_title = glob.glob(path_title + "/*.txt")

    for (fc, ft) in zip(all_content, all_title):
        c = open(fc)
        t = open(ft)
        item = pd.DataFrame({'type':['reliable'],
                             'title':[t.read()],
                             'content':[c.read()]})
                                    
        df = df.append(item)

    title_features, body_features = parse_features(df, title_features, body_features)

    print(title_features)
    print(body_features)

    title_features.to_pickle('FakeNewsAnalysis/Data/title_features_orig.pkl')
    body_features.to_pickle('FakeNewsAnalysis/Data/body_features_orig.pkl') 


if __name__ == "__main__":
    parse_full_dataset()
    #parse_orig_dataset()
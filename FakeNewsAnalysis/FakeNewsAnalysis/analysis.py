import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import matplotlib.dates as md

import sklearn

from textblob import TextBlob, Word, Blobber
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

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
import datetime

import sys
import csv
import ctypes
csv.field_size_limit(int(ctypes.c_ulong(-1).value // 2))

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
    return len(TextBlob(text).words)

def calc_ttr(text):
    return LexicalRichness(text).ttr

def count_nouns(text):
    tags = TextBlob(text).tags
    tags = [i[1] for i in tags]
    return sum(map(lambda x : 1 if 'NN' in x else 0, tags))/len(tags) 

    #tag_count = {value: len(list(freq)) for value, freq in groupby(sorted(tags))}
    #try:
    #    return tag_count['NN']
    #except:
    #    return 0

def count_quotes(text):
    return sum(map(lambda x : 1 if '"' in x else 0, text))/len(text) 

def parse_features(data, title_features, body_features):
    new_body_features = pd.DataFrame({'type':data['type'],
                                      'WC':data['content'].map(count_words),
                                      'TTR':data['content'].map(calc_ttr),
                                      'NN':data['content'].map(count_nouns),
                                      'quote':data['content'].map(count_quotes)})
    #print(new_body_features)
    body_features = body_features.append(new_body_features)
    return title_features, body_features
    #body_features = body_features.reset_index(drop=True)
    #body_features['type'] = body_features['type'].append(data['type'])
    #body_features['WC'] = body_features['WC'].append(data['content'].map(count_words))

def plot_features(body_features, title_features):
    # ['type', 'WC', 'TTR', 'NN', 'quote']


    fake_data = body_features[body_features.type.isin(['fake'])]
    reliable_data = body_features[body_features.type.isin(['reliable'])]

    sns.distplot(fake_data['WC'], hist = True, rug = False, color = 'r')
    sns.distplot(reliable_data['WC'], hist = True, rug = False, color = 'b')  

    plt.figure()
    sns.distplot(body_features[body_features.type.isin(['fake'])].TTR.values, hist = True, rug = False, color = 'r')
    sns.distplot(body_features[body_features.type.isin(['reliable'])].TTR.values, hist = True, rug = False, color = 'b')  

    plt.figure()
    sns.distplot(body_features[body_features.type.isin(['fake'])].NN.values, hist = True, rug = False, color = 'r')
    sns.distplot(body_features[body_features.type.isin(['reliable'])].NN.values, hist = True, rug = False, color = 'b')  

    plt.figure()
    sns.distplot(body_features[body_features.type.isin(['fake'])].quote.values, hist = True, rug = False, color = 'r')
    sns.distplot(body_features[body_features.type.isin(['reliable'])].quote.values, hist = True, rug = False, color = 'b')  
    

if __name__ == "__main__":

    print("start")

    #filename = '../Data/news_full.csv'
    filename = '../Data/news_sample.csv'
    #filename = 'FakeNewsAnalysis/Data/news_sample.csv'

    sample_size = 1  # up to 1

    n_rows = 250
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
        chunk = chunk[chunk.type.isin(['fake', 'reliable'])]
        chunk = chunk.sample(frac=sample_size)
        iteration_count = iteration_count+1
        print('Running iteration: ', iteration_count, "out of: ", int(np.ceil(n_rows/chunk_size)))
        title_features, body_features = parse_features(chunk, title_features, body_features)

    plot_features(body_features, title_features)
    plt.show()
    end = time.time()
    print("total time: ", end - start)

#Full dataset
#Elapsed time  806.0511202812195
#total rows:  84999000000.0
#total time:  806.468124628067
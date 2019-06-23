import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import matplotlib.dates as md

import sklearn

from textblob import TextBlob, Word, Blobber

import pandas as pd
import numpy as np

import plotly
plotly.tools.set_credentials_file(username='feupos', api_key='zplCPm3MX3jp55lj7sMz')
#Plotly Tools
from plotly.offline import init_notebook_mode, iplot
#init_notebook_mode(connected=False)

import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

import time
import datetime

import sys
import csv
import ctypes
csv.field_size_limit(int(ctypes.c_ulong(-1).value // 2))

from multiprocessing.pool import ThreadPool
pool = ThreadPool(20)  # However many you wish to run in parallel

# initialize data structures
start = time.time()
iteration_count = 0
top_words = pd.DataFrame(columns = ['text' , 'count'])
top_authors = pd.DataFrame(columns = ['text' , 'count'])
text_length = pd.DataFrame(columns = ['length'])
word_count = pd.DataFrame(columns = ['count'])
source_date = pd.DataFrame(columns = ['date'])
text_polarity = pd.DataFrame(columns = ['polarity'])
text_type = pd.DataFrame(columns = ['type'])

def preprocess_data(data):
    data.fillna(value=' ') 
    data.dropna()
    data.mask(data.eq('None')).dropna()
    data.mask(data.astype(str).eq('None')).dropna()

def explore_data(data):
    global iteration_count
    iteration_count = iteration_count+1
    print('Runnin iteration: ', iteration_count)
    print('Elapsed time ', time.time() - start)
    preprocess_data(data)  
    explore_dates(data)
    explore_length(data)
    explore_word_count(data)
    explore_polarity(data)
    explore_types(data)
    explore_most_used_words(data)
    explore_authors(data)


def plot_data():

    source_date['date'].iplot(
    kind='hist',
    bins=10,
    xTitle='date',
    linecolor='black',
    yTitle='count',
    title='Date Distribution')

    text_length['length'].iplot(
    kind='hist',
    bins=50,
    xTitle='text length',
    linecolor='black',
    yTitle='count',
    title='Text Length Distribution')

    word_count['count'].iplot(
    kind='hist',
    bins=50,
    xTitle='word count',
    linecolor='black',
    yTitle='count',
    title='Word Count Distribution')

    text_polarity['polarity'].iplot(
    kind='hist',
    bins=50,
    xTitle='polarity',
    linecolor='black',
    yTitle='count',
    title='Sentiment Polarity Distribution')

    text_type['type'].iplot(
    kind='hist',
    xTitle='type',
    linecolor='black',
    yTitle='count',
    title='Content Type Distribution')

    top_words.groupby('text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 100 words')

    top_authors.groupby('text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 100 authors')

def str_to_date(date_str):
    date = datetime.datetime.now()
    try:
        date = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S.%f")
    except:
        print("invalid date format")
    return date

def explore_dates(data):
    try:
        result = pd.DataFrame()
        result['date'] = data['scraped_at']#.map(str_to_date)
        global source_date
        source_date = source_date.append(result)  
    except:
        pass

def explore_length(data):
    try:
        result = pd.DataFrame()
        result['length'] = data['content'].astype(str).apply(len)
        global text_length
        text_length = text_length.append(result)
    except:
        pass

def explore_word_count(data):
    try:
        result = pd.DataFrame()
        result['count'] = data['content'].apply(lambda x: len(str(x).split()))
        global word_count
        word_count = word_count.append(result)
    except:
        pass

def get_polarity(text):
    polarity = 0    
    try:
        polarity = TextBlob(text).sentiment.polarity
    except:
        print("invalid content for polarity")
    return polarity

def explore_polarity(data):
    try:
        result = pd.DataFrame()
        result['polarity'] = data['content'].map(get_polarity)
        global text_polarity
        text_polarity = text_polarity.append(result)
    except:
        pass

def explore_most_used_words(data):
    try:
        vectorizer = sklearn.feature_extraction.text.CountVectorizer(stop_words='english')
        vec = vectorizer.fit_transform(data['content'].replace(np.nan, ' '))
        sum_words = vec.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
        global top_words
        top_words = top_words.append(pd.DataFrame(words_freq[:100], columns = ['text' , 'count']))
        top_words.groupby('text',as_index=False)['count'].sum()
    except:
        pass

def explore_authors(data):
    try:
        vectorizer = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(2, 6) , stop_words='english')
        vec = vectorizer.fit_transform(data['authors'].replace(np.nan, 'None'))
        sum_words = vec.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
        global top_authors
        top_authors = top_authors.append(pd.DataFrame(words_freq[:100], columns = ['text' , 'count']))
        top_authors.groupby('text',as_index=False)['count'].sum()
    except:
        pass

def explore_types(data):
    try:
        result = pd.DataFrame()
        result['type'] = data['type']
        global text_type
        text_type = text_type['type'].append(result)
    except:
        pass

if __name__ == "__main__":
    print("start")
    df_chunk = pd.read_csv('../Data/news_full.csv', chunksize = 10000, nrows = 1000000, engine='python', skip_blank_lines=True, header = 0, error_bad_lines = False)#, date_parser=dateparse, parse_dates=['scraped_at'])
    #df_chunk = pd.read_csv('../Data/news_sample.csv', chunksize = 10000, nrows = 1000000, engine='python', skip_blank_lines=True, header = 0, error_bad_lines = False)#, date_parser=dateparse, parse_dates=['scraped_at'])

    chunk_list = []
    n = 0
    for chunk in df_chunk:
        pool.apply_async(explore_data, (chunk,))

    pool.close()
    pool.join()          
        #chunk_list.append(chunk_out)

    #data_out = pd.concat(chunk_list)

    plot_data()

    end = time.time()
    print("total time: ", end - start)

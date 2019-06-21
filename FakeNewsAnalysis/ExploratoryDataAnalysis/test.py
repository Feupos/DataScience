
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

top_words = pd.DataFrame(columns = ['text' , 'count'])
top_authors = pd.DataFrame(columns = ['text' , 'count'])

def preprocess_data(data):
    data.fillna(value=' ') 
    data.dropna()
    data.mask(data.eq('None')).dropna()
    data.mask(data.astype(str).eq('None')).dropna()

def explore_data(data):
    data_out = pd.DataFrame()
    data_out['dates'] = explore_dates(data)
    data_out['length'] = explore_length(data)
    data_out['word_count'] = explore_word_count(data)
    data_out['polarity'] = explore_polarity(data)
    data_out['types'] = explore_types(data)

    explore_most_used_words(data)
    explore_authors(data)

    return data_out

def plot_data(data):

    data['dates'].iplot(
    kind='hist',
    bins=10,
    xTitle='date',
    linecolor='black',
    yTitle='count',
    title='Date Distribution')

    data['length'].iplot(
    kind='hist',
    bins=10,
    xTitle='text length',
    linecolor='black',
    yTitle='count',
    title='Text Length Distribution')

    data['word_count'].iplot(
    kind='hist',
    bins=10,
    xTitle='word count',
    linecolor='black',
    yTitle='count',
    title='Word Count Distribution')

    data['polarity'].iplot(
    kind='hist',
    bins=20,
    xTitle='polarity',
    linecolor='black',
    yTitle='count',
    title='Sentiment Polarity Distribution')

    data['type'].iplot(
    kind='hist',
    xTitle='type',
    linecolor='black',
    yTitle='count',
    title='Content Type Distribution')

    top_words.groupby('text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 100 words')

    top_authors.groupby('text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 100 authors')

def explore_dates(data):
    ret = data['scraped_at'].map(lambda date_str: datetime.datetime.strptime(str(date_str), "%Y-%m-%d %H:%M:%S.%f"))
    return ret


def explore_length(data):
    ret = data['content'].astype(str).apply(len)
    return ret



def explore_word_count(data):
    return data['content'].apply(lambda x: len(str(x).split()))
    fig, ax = plt.subplots(1, 1, sharey=True, tight_layout=True)
    ax.hist(data['word_count'], bins=10)

    data['word_count'].iplot(
    kind='hist',
    bins=10,
    xTitle='word count',
    linecolor='black',
    yTitle='count',
    title='Word Count Distribution')

def explore_polarity(data):
    return data['content'].map(lambda text: TextBlob(text).sentiment.polarity)
    fig, ax = plt.subplots(1, 1, sharey=True, tight_layout=True)
    ax.hist(data['polarity'], bins=20)

    data['polarity'].iplot(
    kind='hist',
    bins=20,
    xTitle='polarity',
    linecolor='black',
    yTitle='count',
    title='Sentiment Polarity Distribution')

def explore_most_used_words(data):
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(stop_words='english')
    vec = vectorizer.fit_transform(data['content'].replace(np.nan, ' '))
    sum_words = vec.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    top_words.append(pd.DataFrame(words_freq[:100], columns = ['text' , 'count']))
    top_words.groupby('text',as_index=False)['count'].sum()

def explore_authors(data):
    return None
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(stop_words='english')
    vec = vectorizer.fit_transform(data['authors'].replace(np.nan, 'None'))
    sum_words = vec.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    top_authors.append(pd.DataFrame(words_freq[:100], columns = ['text' , 'count']))
    top_authors.groupby('text',as_index=False)['count'].sum()

def explore_types(data):
    return data['type']
    fig, ax = plt.subplots(1, 1, sharey=True, tight_layout=True)
    ax.hist(data['type'])

    data['type'].iplot(
    kind='hist',
    xTitle='type',
    linecolor='black',
    yTitle='count',
    title='Content Type Distribution')
    pass

if __name__ == "__main__":

    start = time.time()
    print("start")
    

    #data_set = load_data('../Data/news_sample.csv')
    #data_set = load_data('../Data/news_full.csv')

    #df_chunk = pd.read_csv(r'../Data/news_sample.csv', chunksize=10000)
    df_chunk = pd.read_csv(r'../Data/news_full.csv', chunksize=10000, low_memory= False)

    chunk_list = []
    n = 0
    for chunk in df_chunk:
        n = n+1
        print("Processing chunk ", n )
        end = time.time()
        print("elapsed time: ", end - start)
        preprocess_data(chunk)
        chunk_out = explore_data(chunk)
        chunk_list.append(chunk_out)

    data_out = pd.concat(chunk_list)

    plot_data(data_out)

    end = time.time()
    print("total time: ", end - start)

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

def load_data(file_name):
    raw_data = pd.read_csv(file_name)
    processed_data = preprocess_data(raw_data)
    return processed_data

def preprocess_data(raw_data):
    processed_data = raw_data
    return processed_data

def explore_data(data):
    explore_dates(data)
    explore_length(data)
    explore_word_count(data)
    explore_polarity(data)
    explore_most_used_words(data)
    explore_types(data)
    # TODO: fix this function
    #explore_authors(data)

    #plt.show()
    

def explore_dates(data):
    data['datetime'] = data['scraped_at'].map(lambda date_str: datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S.%f"))
    fig, ax = plt.subplots(1, 1, sharey=True, tight_layout=True)

    # We can set the number of bins with the `bins` kwarg
    ax.hist(data['datetime'], bins=10)
    xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)

    data['scraped_at'].iplot(
    kind='hist',
    bins=10,
    xTitle='date',
    linecolor='black',
    yTitle='count',
    title='Date Distribution')    

def explore_length(data):
    data['length'] = data['content'].astype(str).apply(len)
    fig, ax = plt.subplots(1, 1, sharey=True, tight_layout=True)
    ax.hist(data['length'], bins=10)

    data['length'].iplot(
    kind='hist',
    bins=10,
    xTitle='text length',
    linecolor='black',
    yTitle='count',
    title='Text Length Distribution')


def explore_word_count(data):
    data['word_count'] = data['content'].apply(lambda x: len(str(x).split()))
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
    data['polarity'] = data['content'].map(lambda text: TextBlob(text).sentiment.polarity)
    fig, ax = plt.subplots(1, 1, sharey=True, tight_layout=True)
    ax.hist(data['polarity'], bins=20)

    data['polarity'].iplot(
    kind='hist',
    bins=20,
    xTitle='polarity',
    linecolor='black',
    yTitle='count',
    title='Sentiment Polarity Distribution')

# adapted from https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a    
def explore_most_used_words(data):
    vec = sklearn.feature_extraction.text.CountVectorizer(stop_words='english').fit(data['content'].replace(np.nan, ' ', inplace=True))
    bag_of_words = vec.transform(data['content'])
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)

    df1 = pd.DataFrame(words_freq[:100], columns = ['Text' , 'count'])

    df1.groupby('Text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 100 words')

# adapted from https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a    
def explore_authors(data):
    vec = sklearn.feature_extraction.text.CountVectorizer(stop_words='english').fit(data['authors'].replace(np.nan, 'None', inplace=True))
    bag_of_words = vec.transform(data['authors'])
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)

    df1 = pd.DataFrame(words_freq[:100], columns = ['Text' , 'count'])

    df1.groupby('Text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 100 authors')

def explore_types(data):
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
    #data_set = load_data('../Data/news_sample.csv')
    data_set = load_data('../Data/news_full.csv')
    explore_data(data_set)

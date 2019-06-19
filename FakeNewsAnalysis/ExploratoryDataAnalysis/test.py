
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import matplotlib.dates as md

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
    raw_data = pd.read_csv(file_name, nrows=1000)
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
    #plt.show()
    

def explore_dates(data):
    data['datetime'] = data['scraped_at'].map(lambda date_str: datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S.%f"))
    fig, ax = plt.subplots(1, 1, sharey=True, tight_layout=True)

    # We can set the number of bins with the `bins` kwarg
    ax.hist(data['datetime'])
    xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)
    #plt.show()

    data['datetime'].iplot(
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
    #plt.show()


def explore_word_count(data):
    data['word_count'] = data['content'].apply(lambda x: len(str(x).split()))
    fig, ax = plt.subplots(1, 1, sharey=True, tight_layout=True)
    ax.hist(data['word_count'], bins=10)
    #plt.show()

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
    #plt.show()

    data['polarity'].iplot(
    kind='hist',
    bins=10,
    xTitle='polarity',
    linecolor='black',
    yTitle='count',
    title='Sentiment Polarity Distribution')

    #iplot(plot)



if __name__ == "__main__":
    #data_set = load_data('../Data/news_sample.csv')
    data_set = load_data('../Data/news_full.csv')
    explore_data(data_set)

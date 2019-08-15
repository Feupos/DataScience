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

def plot_polarity(data):

    reliable = data.loc[data['type'] == 'reliable']
    fake = data.loc[data['type'] == 'fake']

    plt.figure()
    sns.distplot( fake['polarity'], color="red", label="Fake", bins = np.arange(-1,1,0.05), norm_hist  = True)
    sns.distplot( reliable['polarity'] , color="skyblue", label="Reliable", bins = np.arange(-1,1.1,0.05), norm_hist  = True)
    plt.legend()
    plt.xlabel("Polarity")
    plt.ylabel("Count")
    plt.title("Text Polarity")
    plt.tight_layout()
    plt.savefig('results/polarity.png')

    plt.figure()

    sns.violinplot(x="type", y="polarity", data=data,
                        inner=None, color=".8")
    # Show each observation with a scatterplot
    sns.stripplot(x="type", y="polarity", data=data, 
                dodge=True, jitter=True,
                alpha=.25, zorder=1)
    # Show the conditional means
    sns.pointplot(x="type", y="polarity", data=data, 
                dodge=.532, join=False, palette="dark",
                markers="d", scale=.75, ci=None)

    plt.title("Text Polarity by Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/type_polarity.png')

    plt.figure()

    top_words = data['top_words'].value_counts().index.values[:10]

    top_words_polarity = data[data.top_words.isin(top_words)]

    sns.violinplot(x="top_words", y="polarity", data=top_words_polarity,
                        inner=None, color=".8")
    # Show each observation with a scatterplot
    sns.stripplot(x="top_words", y="polarity", data=top_words_polarity, 
                dodge=True, jitter=True,
                alpha=.25, zorder=1)
    # Show the conditional means
    sns.pointplot(x="top_words", y="polarity", data=top_words_polarity, 
                dodge=.532, join=False, palette="dark",
                markers="d", scale=.75, ci=None)

    plt.title("Text Polarity by Word")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/word_polarity.png')
    



def plot_word_freq_compare(fake, reliable, n):

    plt.figure()
    sns.barplot(x="count", y="word", data=fake[:n], color="red")
    plt.xlabel("Count")
    plt.ylabel("Word")
    plt.title("Top Words - Fake")
    plt.tight_layout()
    plt.savefig('results/top_words_fake.png')
    plt.figure()
    sns.barplot(x="count", y="word", data=reliable[:n], color="skyblue")
    plt.xlabel("Count")
    plt.ylabel("Word")
    plt.title("Top Words - Reliable")
    plt.tight_layout()
    plt.savefig('results/top_words_reliable.png')

def explore_words(data, content_type, ngram_min = 1, ngram_max = 1):
    try:
        df = data.loc[data['type'] == content_type]
        print(len(df.index), 'entries for ', content_type)
        vectorizer = sklearn.feature_extraction.text.CountVectorizer(stop_words='english', ngram_range=(ngram_min, ngram_max))
        vec = vectorizer.fit_transform(df['content'])
        sum_words = vec.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
        return pd.DataFrame(words_freq, columns = ['word', 'count'])
    except Exception as e: 
        print(e)
        return pd.DataFrame(None, columns = ['word', 'count'])

def explore_polarity(data):
    try:
        result = pd.DataFrame()
        result['polarity'] = data['content'].apply(lambda text: TextBlob(text).sentiment.polarity)
        result['type'] = data['type']
        vectorizer = sklearn.feature_extraction.text.CountVectorizer(stop_words='english', ngram_range=(1, 1))
        top_words = []
        for entry in data['content']:
            vec = vectorizer.fit_transform([entry])
            sum_words = vec.sum(axis=0)
            words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
            words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
            word = words_freq[0][0]
            top_words = top_words + [word] #[list(zip(*words_freq))[0]]# [words_freq[:10][0]]

        result['top_words'] = top_words
        return result
    except Exception as e: 
        print(e)
        return pd.DataFrame(None, columns = ['type', 'polarity', 'top_words'])


    

def preprocess_data(data):

    data_out = pd.DataFrame()
    data_out = data[['type','content']]
    data_out.dropna(inplace=True)
    return data_out

if __name__ == "__main__":

    print("start")

    #filename = '../Data/news_full.csv'
    #filename = '../Data/news_sample.csv'
    filename = 'FakeNewsAnalysis/Data/news_sample.csv'

    sample_size = 1  # up to 1
    #sample_size = 1

    # df_chunk = pd.read_csv( filename, chunksize = 100000, header = 0,
    #                         engine='python', skip_blank_lines=True,  error_bad_lines = False,
    #                         skiprows = lambda i: i>0 and random.random() > sample_size)
    # row_count = 0
    # for chunk in df_chunk:
    #     row_count = row_count + len(chunk.index)

    # print('sample size:',sample_size*100,'%')
    # print('processed rows: ',row_count)
    # print('total rows: ',row_count/sample_size)
    n_rows = 50
    chunk_size = 1000

    df_chunk = pd.read_csv( filename, chunksize = chunk_size, header = 0, nrows = n_rows,
                            engine='python', skip_blank_lines=True,  error_bad_lines = False)
                            #skiprows=lambda i: i>0 and random.random() > sample_size)

    dataset = pd.DataFrame()
    polarity = pd.DataFrame()
    words_freq_fake = pd.DataFrame()
    words_freq_reliable = pd.DataFrame()
    trump_count = 0
    hillary_count = 0
    iteration_count = 0
    total_count = 0
    for chunk in df_chunk:
        chunk = chunk.sample(frac=sample_size)
        
        for entry in chunk['content']:
            total_count = total_count+1
            #print("!-----------------")
            #print(entry)
            if (entry.find('Trump') > 0):
                #print(entry[entry.find('Trump')-20:entry.find('Trump')+20])
                trump_count = trump_count+1
            if (entry.find('Hillary') > 0):
                #print(entry[entry.find('Hillary')-20:entry.find('Hillary')+20])
                hillary_count = hillary_count+1
            #print("------------------!")
        print(total_count, trump_count, hillary_count)

        iteration_count = iteration_count+1
        print('Running iteration: ', iteration_count, "out of: ", int(np.ceil(n_rows/chunk_size)))
        it_time = time.time()       
        polarity = polarity.append(explore_polarity(chunk),ignore_index = True)
        print('Polarity analysis time: ', time.time() - it_time)
        it_time = time.time()
        #words_freq_fake = words_freq_fake.append(explore_words(chunk,'fake',2,2),ignore_index=True)
        print('Word freq (fake) analysis time: ', time.time() - it_time)
        it_time = time.time()   
        #words_freq_reliable = words_freq_reliable.append(explore_words(chunk,'reliable',2,2),ignore_index=True)
        print('Word freq (reliable) analysis time:', time.time() - it_time)  
        print('Elapsed time ', time.time() - start)
    print(polarity)

    plt.figure()

    #words_freq_fake = words_freq_fake.groupby(['word']).sum()[['count']].reset_index()
    #words_freq_fake = words_freq_fake.sort_values(by=['count'], ascending = False)
    #words_freq_reliable = words_freq_reliable.groupby(['word']).sum()[['count']].reset_index()
    #words_freq_reliable = words_freq_reliable.sort_values(by=['count'], ascending = False)
    #plot_polarity(polarity)
    #plot_word_freq_compare(words_freq_fake,words_freq_reliable,20)
    #plt.show()
    print(type(polarity))
    print(polarity)
    sns.boxplot(x = 'polarity', data = polarity,
                whis="range", palette="vlag")
    plt.show()
    end = time.time()
    print("total time: ", end - start)

#Full dataset
#Elapsed time  806.0511202812195
#total rows:  84999000000.0
#total time:  806.468124628067
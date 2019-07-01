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

from multiprocessing.pool import ThreadPool
pool = ThreadPool(20)  # However many you wish to run in parallel

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
    plt.savefig('results/polarity.png')

def plot_word_freq_compare(fake, reliable, n):

    plt.figure()
    sns.barplot(x="count", y="word", data=fake[:n], color="red")
    plt.xlabel("Count")
    plt.ylabel("Word")
    plt.title("Top Words - Fake")
    plt.figure()

    plt.savefig('results/top_words_fake.png')
    sns.barplot(x="count", y="word", data=reliable[:n], color="skyblue")
    plt.xlabel("Count")
    plt.ylabel("Word")
    plt.title("Top Words - Reliable")
    plt.savefig('results/top_words_reliable.png')

    print(reliable[:n])
    print(fake[:n])

def explore_words(data, content_type, ngram_min = 1, ngram_max = 1):

    try:
        df = data.loc[data['type'] == content_type]
        print(len(df.index), 'entries for ', content_type)
        vectorizer = sklearn.feature_extraction.text.CountVectorizer(stop_words='english', ngram_range=(ngram_min, ngram_max))
        #beg = time.time()
        #df['content'] = df['content'].apply(lambda text: ' '.join(word for word in TextBlob(text).words if word not in stop))
        #df['content'] = df['content'].apply(lambda text: ' '. join([word.lemmatize() for word in TextBlob(text).words if word not in stop]))
        #print('time to filter words: ', time.time() - beg)
        vec = vectorizer.fit_transform(df['content'])
        sum_words = vec.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
        return pd.DataFrame(words_freq, columns = ['word' , 'count'])
    except Exception as e: 
        print(e)
        return pd.DataFrame(None, columns = ['word' , 'count'])

def explore_polarity(data):

    result = pd.DataFrame()
    result['polarity'] = data['content'].apply(lambda text: TextBlob(text).sentiment.polarity)
    result['type'] = data['type']
    return result
    

def preprocess_data(data):

    data_out = pd.DataFrame()
    data_out = data[['type','content']]
    data_out.dropna(inplace=True)
    #data_out['content'] = data['content'].apply(lambda text: TextBlob(text))
    return data_out

if __name__ == "__main__":

    print("start")

    filename = '../Data/news_full.csv'
    #filename = '../Data/news_sample.csv'

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

    n_rows = 10000
    chunk_size = 10000
    df_chunk = pd.read_csv( filename, chunksize = chunk_size, header = 0, nrows = n_rows,
                            engine='python', skip_blank_lines=True,  error_bad_lines = False,
                            skiprows = lambda i: i>0 and random.random() > sample_size )

    dataset = pd.DataFrame()
    polarity = pd.DataFrame()
    words_freq_fake = pd.DataFrame()
    words_freq_reliable = pd.DataFrame()

    iteration_count = 0
    for chunk in df_chunk:
        iteration_count = iteration_count+1
        print('Running iteration: ', iteration_count, "out of: ", int(np.ceil(n_rows*sample_size/chunk_size)))
        it_time = time.time()       
        polarity = polarity.append(explore_polarity(chunk),ignore_index = True)
        print('Polarity analysis time: ', time.time() - it_time)
        it_time = time.time()   
        words_freq_fake = words_freq_fake.append(explore_words(chunk,'fake',3,3),ignore_index = True)
        print('Word freq (fake) analysis time: ', time.time() - it_time)
        it_time = time.time()   
        words_freq_reliable = words_freq_reliable.append(explore_words(chunk,'reliable',3,3),ignore_index = True)
        print('Word freq (reliable) analysis time:', time.time() - it_time)  
        print('Elapsed time ', time.time() - start)

    words_freq_fake.groupby('word',as_index=False)['count'].sum()
    words_freq_reliable.groupby('word',as_index=False)['count'].sum()
    plot_polarity(polarity)
    plot_word_freq_compare(words_freq_fake,words_freq_reliable,20)
    plt.show()

    end = time.time()
    print("total time: ", end - start)

#Full dataset
#Elapsed time  806.0511202812195
#total rows:  84999000000.0
#total time:  806.468124628067
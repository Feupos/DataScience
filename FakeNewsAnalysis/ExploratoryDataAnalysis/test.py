import pandas as pd
import matplotlib as mpl
from textblob import TextBlob, Word, Blobber

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

def explore_dates(data):
    dates_str = data['scraped_at']
    print(dates_str)   

if __name__ == "__main__":
    data_set = load_data('./news_sample.csv')
    explore_data(data_set)

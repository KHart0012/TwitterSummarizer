# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 09:54:46 2019

@author: s526939
"""

import pickle
import nltk
import numpy
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

stop_words = set(stopwords.words('english'))
for punct in ",.'?/:;~*&%!()$+=-_<>“”’":
    stop_words.add(punct)
tweet_tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
trend_dict = {} # Contains the number which corisponds to the file the tweets are located in.
summary_dict = {} # Contains all the current summaries of cached trends
trend_names = []
MAX_TRENDS = 24

def initialize_data():
    create_trend_dictionary()
    #create_summary_dictionary()

def create_trend_dictionary():
    for i in range(1, MAX_TRENDS):
        with open('pickles/trend' + str(i) + 'names.pkl', 'rb') as f:
            names = pickle.load(f)
        trend_names.extend(names)
        for trend in set(names):
            if trend in list(trend_dict.keys()):
                trend_dict[trend].append(i)
            else:
                trend_dict[trend] = []
                trend_dict[trend].append(i)
            summary_dict[trend] = ''

def create_summary_dictionary():
    for trend in summary_dict.keys():
        summary_dict[trend] = summarize_trend(trend)

def summarize_sample_data():
    summ_dict = {}
    with open('pickles/trendtesttweets.pkl', 'rb') as f:
        tweet_dict = pickle.load(f)
    for trend in tweet_dict.keys():
        summ_dict[trend] = summarize_tweets(tweet_dict[trend])
    return summ_dict

# Given a trend, it will grab the stored tweets and summarize it
def summarize_trend(trend, N=12):
    tweets = grab_tweets(trend)
    sentences = list(set([tweet.full_text for tweet in tweets]))
    
    words = list()
    for sent in sentences:
        words.extend(map(lambda x: x.lower(), tweet_tokenizer.tokenize(sent)))
        
    freqDist = nltk.FreqDist(words)
    sorted_terms = sorted(freqDist.items(), key=lambda x: x[1], reverse=True)
    n_most_common = [word[0] for word in sorted_terms if word[0] not in stop_words][:N]
    
    return summarize(sentences, n_most_common)

# Given a list of tweets it will summarize them
def summarize_tweets(tweets, N=12):
    sentences = list(set([tweet.full_text for tweet in tweets]))
    
    words = list()
    for sent in sentences:
        words.extend(map(lambda x: x.lower(), tweet_tokenizer.tokenize(sent)))
        
    freqDist = nltk.FreqDist(words)
    sorted_terms = sorted(freqDist.items(), key=lambda x: x[1], reverse=True)
    n_most_common = [word[0] for word in sorted_terms if word[0] not in stop_words][:N]
    
    return summarize(sentences, n_most_common)


def cluster_score(cluster):
    sig_words = len(cluster)
    total_words = cluster[-1] - cluster[0] + 1
    return sig_words ** 2 / total_words

def score_sentences(sentences, important_words, CLUSTER_THRESH=5):
    #nltk.download('punkt')
    scores = []
    
    for sent in map(tweet_tokenizer.tokenize, sentences):
        word_idx = []
        for word in important_words:
            if word in sent:
                word_idx.append(sent.index(word))
        word_idx.sort()
        if len(word_idx) > 0:
            clusters = []
            current_cluster = [word_idx[0]]
            for idx in word_idx[1:]:
                if idx - word_idx[-1] < CLUSTER_THRESH:
                    current_cluster.append(idx)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [idx]
            clusters.append(current_cluster)
            scores.append(max(map(cluster_score, clusters)))
        else:
            scores.append(0)
    return scores

def summarize(sentences, important_words, CTHRESH=5, N=3):
    # More flexible way to summarize
    scores = score_sentences(sentences, important_words, CTHRESH)
    avg = numpy.mean(scores)
    std_dev = numpy.std(scores)
    score_thresh = avg + 0.5 * std_dev
    mean_scored = [t[0] for t in enumerate(scores) if t[1] > score_thresh][:N]
    
    return clean_summary(' '.join(sentences[i] for i in mean_scored))

def grab_tweets(trend):
    tweet_list = []
    for file_num in trend_dict[trend]:
        with open('pickles/trend' + str(file_num) + 'tweets.pkl', 'rb') as f:
            temp_tweets = pickle.load(f)
        tweet_list.extend(temp_tweets[trend])
    return tweet_list

def tokenize_tweet(tweet):
    return remove_hypers(tweet_tokenizer.tokenize(tweet.full_text))

def remove_hypers(tokens):
    return list(filter(lambda x: not x.startswith('http'), tokens))

def remove_stopwords(tokens):
    return list(filter(lambda x: not x in stop_words, tokens))

def remove_punctuation(tokens):
    return list(filter(lambda x: not x in ",.'?/:;~*&%!()$+=-_<>“”’", tokens))

def clean_summary(summary):
    tokens = remove_punctuation(remove_hypers(tweet_tokenizer.tokenize(summary)))
    return ' '.join(token for token in tokens)     

def trend_wordcloud(trend):
    with open('pickles/' + trend + '_text.pkl', 'rb') as f:
        text = pickle.load(f)
    display_wordcloud(WordCloud().generate(text))

def test_wordcloud(trend):
    display_wordcloud(generate_wordcloud(trend))

def display_wordcloud(wc):
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def generate_wordcloud(trend):
    tweets = grab_tweets(trend)
    texts = list()
    for tweet in tweets:
        texts.extend(map(lambda x: x.lower(), remove_stopwords(tokenize_tweet(tweet))))
    
    text = ' '.join(token for token in texts)
    with open('pickles/' + trend + '_text.pkl', 'wb') as f:
        pickle.dump(text, f)
    return WordCloud().generate(text)


# Main
#initialize_data()
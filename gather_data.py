# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:09:23 2019

@author: Kevin Hart | s526939

Non-Cursor API Calls
 - Trends 15 Tweets 150: ~50 Calls
 - Trends 30 Tweets 150: ~78 Calls
 - Trends 50 Tweets 150: ~134 Calls
"""

import tweepy
import pickle
from twitter_keys import consumer_key, consumer_secret, access_token, access_secret

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth)

max_tweets = 175
num_trends = 10
trend_count = 24

def gather_sample_data():
    trends_us = api.trends_place(23424977)
    trends = []
    trend_queries = []
    tweets_dict = dict()
    
    # Grabs Trends from Twitter
    for trend in trends_us[0]['trends'][:num_trends]:
        trends.append(trend['name'])
        trend_queries.append(trend['query'])
        tweets_dict[str(trend['name'])] = []
    
    # Pickles the trend names
    with open('pickles/trendtestnames.pkl', 'wb') as f:
        pickle.dump(trends, f)

    # Non-cursor way is way more efficient with it's API calls
    for i in range(0, num_trends):
        query = trend_queries[i]
        searched_tweets = []
        last_id = -1
        while len(searched_tweets) < max_tweets:
            count = max_tweets - len(searched_tweets)
            try:
                new_tweets = api.search(q=query, count=count, max_id=str(last_id - 1), tweet_mode='extended')
                if not new_tweets:
                    break
                searched_tweets.extend(new_tweets)
                last_id = new_tweets[-1].id
            except tweepy.TweepError:
                print('Ran out of API calls')
                i = 100
                break
    
        # Grabs the retweeted text because retweets don't have full_text
        try:
            no_retweets = []
            for status in searched_tweets:
                if hasattr(status, 'retweeted_status'):
                    no_retweets.append(status.retweeted_status)
                else:
                    no_retweets.append(status)
            tweets_dict[str(trends[i])].extend(no_retweets)
        except IndexError:
            pass

    # Pickles tweets with trend name as key and tweets as value
    with open('pickles/trendtesttweets.pkl', 'wb') as f:
        pickle.dump(tweets_dict, f)
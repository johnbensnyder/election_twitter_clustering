"""
Created on Tue Dec  6 14:32:12 2016

@author: johnsnyder

This code reads in the csv files with twitter data, and outputs
a single dataset with all tweet text, time, and user data.

"""

import os
import pandas as pd
import re
from datetime import datetime

#os.chdir('/Users/johnsnyder/Documents/Fall_2016/cs505/final_project/to_download/trump_follower_tweets')

#os.listdir()

def get_tweets_by_user(filename):
    tweets = pd.read_csv(filename)
    tweets_by_user = tweets[['created','user_ids','text']]
    #tweets_by_user['text'] = [re.findall(r"#(\w+)", str(s)) for s in tweets_by_user['text']]
    #tweets_by_user = tweets_by_user.loc[[True if len(i)>0 else False for i in tweets_by_user['text']]]
    tweets_by_user['dateTime_created'] = [datetime.strptime(i,'%a %b %d %H:%M:%S +0000 %Y') for i in list(tweets_by_user['created'])]
    tweets_by_user['year'] = [i.year for i in tweets_by_user['dateTime_created']]
    tweets_by_user = tweets_by_user.loc[tweets_by_user['year']==2016]
    tweets_by_user['month'] = [i.month for i in tweets_by_user['dateTime_created']]
    tweets_by_user['day'] = [i.strftime('%j') for i in tweets_by_user['dateTime_created']]    
    tweets_by_user = tweets_by_user[['user_ids','text','month','day']]
    return tweets_by_user

def all_user_tweets(a_folder,target):
    os.chdir(a_folder)
    #tweets_by_user = pd.DataFrame({'user_ids':[],'text':[]})
    user_ids = []
    text = []
    month = []
    day = []
    count = 0
    for a_file in os.listdir():
        if 'tweet_data' in a_file:
            try:
                tweets_by_user = get_tweets_by_user(a_file)
                user_ids.extend(tweets_by_user['user_ids'])
                text.extend(tweets_by_user['text'])
                month.extend(tweets_by_user['month'])
                day.extend(tweets_by_user['day'])
            except:
                continue
            print(count)
            count += 1
    tweets_by_user = pd.DataFrame({'text':text,'user_ids':user_ids,'month':month,'day':day})
    tweets_by_user['target'] = target
    return tweets_by_user

trump_followers = all_user_tweets('/Users/johnsnyder/Documents/Fall_2016/cs505/final_project/to_download/trump_follower_tweets',1)
clinton_followers = all_user_tweets('/Users/johnsnyder/Documents/Fall_2016/cs505/final_project/to_download/clinton_follower_tweets',2)
general_followers = all_user_tweets('/Users/johnsnyder/Documents/Fall_2016/cs505/final_project/to_download/general_follower_tweets',0)


all_tweets = trump_followers.append(clinton_followers)

all_tweets = all_tweets.append(general_followers)
os.chdir('/Users/johnsnyder/Documents/Fall_2016/cs505/final_project')

all_tweets.to_csv('all_tweets_by_day_month.csv')

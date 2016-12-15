
import os
os.chdir('/Users/johnsnyder/Documents/Fall_2016/cs505/final_project')

from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream
import json
import random
import pandas as pd

 
from time import sleep

ACCESS_TOKEN = 
ACCESS_SECRET = 
CONSUMER_KEY = 
CONSUMER_SECRET = 

    
oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)
twitter = Twitter(auth=oauth)

cursor_position = -1

def get_a_group_of_followers(cursor_position):
    aList = twitter.followers.ids(screen_name='realDonaldTrump',
                        count=180,cursor=cursor_position) 
    ids = []
    user_ids = []
    user_name = []
    user_creation = []
    user_description = []
    text = []
    created = []
    source = []                    
    for j in aList['ids']:
            try:
                follower_tweets_seg = twitter.statuses.user_timeline(user_id=j,
                            count=200)
                for i in follower_tweets_seg:
                    try:
                        ids.append(i['id'])
                    except:
                        ids.append['']
                    try:
                        user_ids.append(i['user']['id'])
                    except:
                        user_ids.append['']
                    try:
                        user_name.append(i['user']['name'])
                    except:
                        user_name.append['']
                    try:
                        user_creation.append(i['user']['created_at'])
                    except:
                        user_creation.append['']
                    try:
                        user_description.append(i['user']['description'])
                    except:
                        user_description.append['']
                    try:
                        text.append(i['text'])
                    except:
                        text.append['']
                    try:
                        created.append(i['created_at'])
                    except:
                        created.append['']
                    try:
                        source.append(i['source'])
                    except:
                        source.append['']
                sleep(5) 
            except:
                sleep(5)
                continue
    tweet_data = pd.DataFrame({'ids':ids,'user_ids':user_ids,'user_name':user_name,
             'user_creation':user_creation,'user_description':user_description,
             'text':text,'created':created,'source':source
    })
    tweet_data.to_csv('tweet_data_'+str(cursor_position)+'.csv')

    return aList['next_cursor']

while True:
    cursor_position = get_a_group_of_followers(cursor_position)




# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 12:25:24 2016

@author: johnsnyder
"""


import os
#os.chdir('/Users/johnsnyder/Documents/Fall_2016/cs505/final_project')

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

def get_a_group_of_friends(user,cursor_position):
    aList = twitter.followers.ids(screen_name='realDonaldTrump',
                        count=180,cursor=cursor_position) 
    with open('follower_friends'+str(cursor_position)+'.txt', 'w') as outfile:
        for i in range(15):
            random_follower = random.choice(aList['ids'])
            aList['ids'].remove(random_follower)
            try:
                friends = twitter.friends.ids(user_id = random_follower, count = 200)
                json.dump(friends, outfile)
                outfile.write('\n')
                sleep(60)
            except:
                sleep(60)
                continue
    
    return aList['next_cursor']


while True:
    cursor_position = get_a_group_of_friends(cursor_position)




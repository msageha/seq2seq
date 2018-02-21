from twitter import *
import json
import os
import time
import sys

def getKeyDic(api_name, jsonPath = './api_key.json'):
    with open(jsonPath, 'r') as f:
        keyDic = json.load(f).get(api_name)
    return keyDic

def getUserTweet(auth, name, folder):
  with open(folder, 'a') as f:
    count = 100
    api = Twitter(auth=auth)
    # timelines = api.statuses.user_timeline(screen_name=name, count=count)
    # for timeline in timelines:
    #   f.write(timeline['text']+'\n')
    max_id = 963619750776422399 #指定しないとだめなよう
    while True:
      timelines = api.statuses.user_timeline(screen_name=name, count=count, max_id=max_id)
      if timelines == []:
          print('None')
          break
      max_id = timelines[-1]['id'] - 1
      for timeline in timelines:
            f.write(json.dumps(timeline))
            f.write('\n')
      if len(timelines) < count:
        print(max_id)
        break

if __name__ == '__main__':
    user_id = '3274075003'
    keyDic = getKeyDic('twitter_sm_nz_gk')
    auth = OAuth(keyDic['ACCESS_TOKEN_KEY'], keyDic['ACCESS_TOKEN_SECRET'], keyDic['CONSUMER_KEY'], keyDic['CONSUMER_SECRET'])
    name = '@ms_rinna'
    getUserTweet(auth, name, f'./tweet/{name}.txt')

"""
@ms_rinna : 3274075003
"""
import json
import os
import time
import sys
from requests_oauthlib import OAuth1Session, OAuth1
import requests

def getKeyDic(api_name, jsonPath = './api_key.json'):
    with open(jsonPath, 'r') as f:
        keyDic = json.load(f).get(api_name)
    return keyDic

keyDic = getKeyDic('twitter_msageha')
oauth = OAuth1(keyDic['CONSUMER_KEY'],
        keyDic['CONSUMER_SECRET'],
        keyDic['ACCESS_TOKEN_KEY'],
        keyDic['ACCESS_TOKEN_SECRET'])


def get_tweet(tweet_id):
    response = requests.get(
        f'https://api.twitter.com/1.1/statuses/show.json?id={tweet_id}',
        auth=oauth
    )
    text = ''
    flag = False
    if response.ok:
        tweet = response.json()
        text = tweet['text']
    elif response.status_code == 429:
        flag = True
    return text, flag

def read_tweet_json(file_path):
    dialog = []
    with open(file_path) as f:
        i = 0
        for line in f:
            i += 1
            if i%1000 == 0:
                print(i)
            tweet = json.loads(line)
            if tweet["in_reply_to_status_id"]:
                tweet_id = tweet["in_reply_to_status_id"]
                input_text = tweet['text']
                output_text, flag = get_tweet(tweet_id)
                if output_text != '':
                    dialog.append((input_text, output_text))
                if flag:
                    print(i)
                    break
    return dialog

if __name__ == '__main__':
    file = '2017-11-23.txt'
    dialog = read_tweet_json(f'tweet/{file}')
    with open(f'tweet_dialog/{file}', 'w') as f:
        for i in dialog:
            f.write(f'input: {i[0]}')
            f.write(f'output: {i[1]}')
    
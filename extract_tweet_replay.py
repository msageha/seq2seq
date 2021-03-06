import json
import os
import time
import sys
from requests_oauthlib import OAuth1Session, OAuth1
import requests
import re

def remove_account(text):
    text = re.sub(r'@[!-~a-zA-Z_+.-0-9]+', '', text)
    return text

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

def write_dialog(file_path, dialog):
    with open(f'dialog/{file}', 'a') as f:
        for i in dialog:
            f.write(f'input: {i[0]}\n')
            f.write(f'output: {i[1]}\n')

def read_tweet_json(file):
    dialog = []
    with open(f'tweet/{file}') as f:
        i = 0
        for line in f:
            i += 1
            if i%1000 == 0:
                print(i)
            tweet = json.loads(line)
            if tweet['user']['name'] != 'りんな':
                continue
            if tweet["in_reply_to_status_id"]:
                tweet_id = tweet["in_reply_to_status_id"]
                input_text = tweet['text'].replace('\n', ' ')
                output_text, flag = get_tweet(tweet_id)
                output_text = output_text.replace('\n', ' ')
                input_text = remove_account(input_text)
                output_text = remove_account(output_text)
                if output_text != '':
                    dialog.append((input_text, output_text))
                if flag:
                    print(i)
                    write_dialog(file, dialog)
                    dialog = []
                    print('1min sleep')
                    time.sleep(60)
    return dialog

if __name__ == '__main__':
    for file in os.listdir('./tweet'):
        read_tweet_json(f'{file}')

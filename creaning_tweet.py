import re
from emoticon_dicts import *
from emoji_dicts import *
import os

emoji_dict = {}
emoji_dict.update(smiley_dict)
emoji_dict.update(person_dict)
emoji_dict.update(role_dict)
emoji_dict.update(fantasy_dict)
emoji_dict.update(gesture_activity_dict)
emoji_dict.update(sport_dict)
emoji_dict.update(family_dict)
emoji_dict.update(skintone_body_dict)
emoji_dict.update(emotion_clothing_dict)
emoji_dict.update(animal_nature_dict)
emoji_dict.update(animal_nature_dict)
emoji_dict.update(food_drink_dict)
emoji_dict.update(travel_place_dict)
emoji_dict.update(activity_dict)
emoji_dict.update(object_dict)
emoji_dict.update(symbol_dict)
emoji_dict.update(flag_dict)
emoji_dict_key_sorted = sorted(emoji_dict.keys(), key=len, reverse=True)

emoticon_dict = {}
emoticon_dict.update(smile_dict)
emoticon_dict.update(sweat_dict)
emoticon_dict.update(other_dict)
emoticon_dict.update(sadness_dict)
emoticon_dict.update(displeasure_dict)
emoticon_dict.update(surprise_dict)
emoticon_dict_key_sorted = sorted(emoticon_dict.keys(), key=len, reverse=True)

def remove_url(text):
    text = re.sub(r'http[!-~a-zA-Z_+.-0-9]+', ' ', text)
    return text

def remove_kakko(text):
    text = re.sub(r'\(.+\)', '', text)
    return text

def remove_hash(text):
    text = re.sub(r'#[\S]+', ' ', text)
    return text

def remove_emoji(text):
    for emoji in emoji_dict_key_sorted:
        text = text.replace(emoji, ' ')
    return text

def remove_emoticon(text):
    for emoticon in emoticon_dict_key_sorted:
        text = text.replace(emoticon, ' ')
    return text

def remove_account(text):
    text = re.sub(r'@[!-~a-zA-Z_+.-0-9]+', '', text)
    return text

def main(file):
    with open(f'tweet_dialog/{file}') as f:
        with open(f'data/normal/{file}', 'w') as w:
            for line in f:
                text = line
                text = remove_url(text)
                text = remove_emoji(text)
                text = remove_emoticon(text)
                text = remove_hash(text)
                text = remove_account(text)
                input_text = text.strip()
                text = f.readline()
                text = remove_url(text)
                text = remove_emoji(text)
                text = remove_emoticon(text)
                text = remove_hash(text)
                text = remove_account(text)
                output_text = text.strip()
                input_text = input_text.replace('input:', '')
                output_text = output_text.replace('output:', '')
                if input_text.strip() == '' or output_text.strip() == '':
                    continue
                w.write(f'input: {output_text}\n')
                w.write(f'output: {input_text}\n')

if __name__ == '__main__':
    for file in os.listdir('tweet_dialog'):
        main(file)

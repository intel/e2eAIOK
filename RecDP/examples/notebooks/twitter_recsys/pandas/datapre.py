#!/env/bin/python

import os, sys
import pandas as pd
#import modin.experimental.pandas as pd
#import pandas as pd
import numpy as np 
import hashlib
from timeit import default_timer as timer
from collections import Counter
import time 
from pathlib import Path
import shutil
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

target_list = [
    'reply_timestamp',
    'retweet_timestamp',
    'retweet_with_comment_timestamp',
    'like_timestamp'
    ]
indexlist = ["engaging_user_id","tweet_id"]
stage1_features_list = [
    'engaged_with_user_follower_count',
    'engaged_with_user_following_count',
    'engaged_with_user_is_verified',
    'engaging_user_follower_count',
    'engaging_user_following_count',
    'engaging_user_is_verified',
    'has_photo',
    'has_video',
    'has_gif',
    'a_ff_rate',
    'b_ff_rate', 
    'dt_hour',
    'dt_dow',
    'has_mention',  
    'mentioned_bucket_id',    
    'mentioned_count',    
    'most_used_word_bucket_id',
    'second_used_word_bucket_id',
    'TE_tweet_type_reply_timestamp',
    'TE_tweet_type_retweet_timestamp',
    'TE_dt_dow_retweet_timestamp',
    'TE_most_used_word_bucket_id_reply_timestamp',
    'TE_most_used_word_bucket_id_retweet_timestamp',
    'TE_most_used_word_bucket_id_retweet_with_comment_timestamp',
    'TE_most_used_word_bucket_id_like_timestamp',
    'TE_second_used_word_bucket_id_reply_timestamp',
    'TE_second_used_word_bucket_id_retweet_timestamp',
    'TE_second_used_word_bucket_id_retweet_with_comment_timestamp',
    'TE_second_used_word_bucket_id_like_timestamp',
    'TE_mentioned_bucket_id_retweet_timestamp',
    'TE_mentioned_bucket_id_retweet_with_comment_timestamp',
    'TE_mentioned_bucket_id_like_timestamp',
    'TE_mentioned_bucket_id_reply_timestamp',
    'TE_language_reply_timestamp',
    'TE_language_retweet_timestamp',
    'TE_language_retweet_with_comment_timestamp',
    'TE_language_like_timestamp',
    'TE_mentioned_count_reply_timestamp',
    'TE_mentioned_count_retweet_timestamp',
    'TE_mentioned_count_retweet_with_comment_timestamp',
    'TE_mentioned_count_like_timestamp',
    'TE_engaged_with_user_id_reply_timestamp',
    'TE_engaged_with_user_id_retweet_timestamp',
    'TE_engaged_with_user_id_retweet_with_comment_timestamp',
    'TE_engaged_with_user_id_like_timestamp',
    'GTE_language_engaged_with_user_id_reply_timestamp',
    'GTE_language_engaged_with_user_id_retweet_timestamp',
    'GTE_language_engaged_with_user_id_retweet_with_comment_timestamp',
    'GTE_language_engaged_with_user_id_like_timestamp',
    'GTE_tweet_type_engaged_with_user_id_reply_timestamp',
    'GTE_tweet_type_engaged_with_user_id_retweet_timestamp',
    'GTE_tweet_type_engaged_with_user_id_retweet_with_comment_timestamp',
    'GTE_tweet_type_engaged_with_user_id_like_timestamp',
    'GTE_has_mention_engaging_user_id_reply_timestamp',
    'GTE_has_mention_engaging_user_id_retweet_timestamp',
    'GTE_has_mention_engaging_user_id_retweet_with_comment_timestamp',
    'GTE_has_mention_engaging_user_id_like_timestamp',
    'GTE_mentioned_bucket_id_engaging_user_id_reply_timestamp',
    'GTE_mentioned_bucket_id_engaging_user_id_retweet_timestamp',
    'GTE_mentioned_bucket_id_engaging_user_id_retweet_with_comment_timestamp',
    'GTE_mentioned_bucket_id_engaging_user_id_like_timestamp',
    'GTE_language_engaging_user_id_reply_timestamp',
    'GTE_language_engaging_user_id_retweet_timestamp',
    'GTE_language_engaging_user_id_retweet_with_comment_timestamp',
    'GTE_language_engaging_user_id_like_timestamp',
    'GTE_tweet_type_engaging_user_id_reply_timestamp',
    'GTE_tweet_type_engaging_user_id_retweet_timestamp',
    'GTE_tweet_type_engaging_user_id_retweet_with_comment_timestamp',
    'GTE_tweet_type_engaging_user_id_like_timestamp',
    'GTE_dt_dow_engaged_with_user_id_reply_timestamp',
    'GTE_dt_dow_engaged_with_user_id_retweet_timestamp',
    'GTE_dt_dow_engaged_with_user_id_retweet_with_comment_timestamp',
    'GTE_dt_dow_engaged_with_user_id_like_timestamp',
    'GTE_mentioned_count_engaging_user_id_reply_timestamp',
    'GTE_mentioned_count_engaging_user_id_retweet_timestamp',
    'GTE_mentioned_count_engaging_user_id_retweet_with_comment_timestamp',
    'GTE_mentioned_count_engaging_user_id_like_timestamp',
    'GTE_dt_hour_engaged_with_user_id_reply_timestamp',
    'GTE_dt_hour_engaged_with_user_id_retweet_timestamp',
    'GTE_dt_hour_engaged_with_user_id_retweet_with_comment_timestamp',
    'GTE_dt_hour_engaged_with_user_id_like_timestamp',
    'GTE_dt_dow_engaging_user_id_reply_timestamp',
    'GTE_dt_dow_engaging_user_id_retweet_timestamp',
    'GTE_dt_dow_engaging_user_id_retweet_with_comment_timestamp',
    'GTE_dt_dow_engaging_user_id_like_timestamp',
    'GTE_dt_hour_engaging_user_id_reply_timestamp',
    'GTE_dt_hour_engaging_user_id_retweet_timestamp',
    'GTE_dt_hour_engaging_user_id_retweet_with_comment_timestamp',
    'GTE_dt_hour_engaging_user_id_like_timestamp',
    "engagee_follows_engager",
    "dt_minute",
    "len_domains",
    "len_hashtags",
    "len_links",
    "TE_tweet_type_retweet_with_comment_timestamp",
    "TE_tweet_type_like_timestamp",
    "GTE_engaged_with_user_id_engaging_user_id_reply_timestamp",
    "GTE_engaged_with_user_id_engaging_user_id_retweet_timestamp",
    "GTE_engaged_with_user_id_engaging_user_id_retweet_with_comment_timestamp",
    "GTE_engaged_with_user_id_engaging_user_id_like_timestamp",
    "GTE_engaged_with_user_id_language_tweet_type_reply_timestamp",
    "GTE_engaged_with_user_id_language_tweet_type_retweet_timestamp",
    "GTE_engaged_with_user_id_language_tweet_type_retweet_with_comment_timestamp",
    "GTE_engaged_with_user_id_language_tweet_type_like_timestamp",
    "GTE_engaging_user_id_language_tweet_type_reply_timestamp",
    "GTE_engaging_user_id_language_tweet_type_retweet_timestamp",
    "GTE_engaging_user_id_language_tweet_type_retweet_with_comment_timestamp",
    "GTE_engaging_user_id_language_tweet_type_like_timestamp",
    "TE_engaging_user_id_reply_timestamp",
    "TE_engaging_user_id_retweet_timestamp",
    "TE_engaging_user_id_retweet_with_comment_timestamp",
    "TE_engaging_user_id_like_timestamp",
    "GTE_language_tweet_type_present_media_reply_timestamp",
    "GTE_language_tweet_type_present_media_retweet_timestamp",
    "GTE_language_tweet_type_present_media_retweet_with_comment_timestamp",
    "GTE_language_tweet_type_present_media_like_timestamp",
    "TE_present_media_reply_timestamp",
    "TE_present_media_retweet_timestamp",
    "TE_present_media_retweet_with_comment_timestamp",
    "TE_present_media_like_timestamp",
    'TE_tw_word0_reply_timestamp', 
    'TE_tw_word0_retweet_timestamp', 
    'TE_tw_word0_retweet_with_comment_timestamp', 
    'TE_tw_word0_like_timestamp',
    "len_media",
    "ab_age_dff",
    "ab_age_rate",
    "ab_fing_rate",
    "ab_fer_rate",
    'GTE_engaging_user_is_verified_tweet_type_reply_timestamp', 
    'GTE_engaging_user_is_verified_tweet_type_retweet_timestamp', 
    'GTE_engaging_user_is_verified_tweet_type_retweet_with_comment_timestamp', 
    'GTE_engaging_user_is_verified_tweet_type_like_timestamp',
    'GTE_present_domains_language_engagee_follows_engager_tweet_type_present_media_engaged_with_user_is_verified_reply_timestamp', 
    'GTE_present_domains_language_engagee_follows_engager_tweet_type_present_media_engaged_with_user_is_verified_retweet_timestamp', 
    'GTE_present_domains_language_engagee_follows_engager_tweet_type_present_media_engaged_with_user_is_verified_retweet_with_comment_timestamp', 
    'GTE_present_domains_language_engagee_follows_engager_tweet_type_present_media_engaged_with_user_is_verified_like_timestamp',
    'GTE_present_media_tweet_type_language_engaged_with_user_is_verified_engaging_user_is_verified_engagee_follows_engager_reply_timestamp', 
    'GTE_present_media_tweet_type_language_engaged_with_user_is_verified_engaging_user_is_verified_engagee_follows_engager_retweet_timestamp', 
    'GTE_present_media_tweet_type_language_engaged_with_user_is_verified_engaging_user_is_verified_engagee_follows_engager_retweet_with_comment_timestamp', 
    'GTE_present_media_tweet_type_language_engaged_with_user_is_verified_engaging_user_is_verified_engagee_follows_engager_like_timestamp',
    'GTE_tw_original_user0_tweet_type_language_reply_timestamp', 
    'GTE_tw_original_user0_tweet_type_language_retweet_timestamp', 
    'GTE_tw_original_user0_tweet_type_language_retweet_with_comment_timestamp', 
    'GTE_tw_original_user0_tweet_type_language_like_timestamp',
    'GTE_tw_original_user1_tweet_type_language_reply_timestamp', 
    'GTE_tw_original_user1_tweet_type_language_retweet_timestamp', 
    'GTE_tw_original_user1_tweet_type_language_retweet_with_comment_timestamp', 
    'GTE_tw_original_user1_tweet_type_language_like_timestamp'
    ]
stage2_features_list = [
    'stage2_TE_engaged_with_user_id_reply_timestamp',
    'stage2_TE_engaged_with_user_id_retweet_timestamp',
    'stage2_TE_engaged_with_user_id_retweet_with_comment_timestamp',
    'stage2_TE_engaged_with_user_id_like_timestamp',
    'stage2_TE_language_reply_timestamp',
    'stage2_TE_language_retweet_timestamp',
    'stage2_TE_language_retweet_with_comment_timestamp',
    'stage2_TE_language_like_timestamp',
    'stage2_TE_dt_dow_retweet_timestamp',
    'stage2_TE_tweet_type_reply_timestamp',
    'stage2_TE_tweet_type_retweet_timestamp',
    'stage2_TE_most_used_word_bucket_id_reply_timestamp',
    'stage2_TE_most_used_word_bucket_id_retweet_timestamp',
    'stage2_TE_most_used_word_bucket_id_retweet_with_comment_timestamp',
    'stage2_TE_most_used_word_bucket_id_like_timestamp',
    'stage2_TE_second_used_word_bucket_id_reply_timestamp',
    'stage2_TE_second_used_word_bucket_id_retweet_timestamp',
    'stage2_TE_second_used_word_bucket_id_retweet_with_comment_timestamp',
    'stage2_TE_second_used_word_bucket_id_like_timestamp',
    'stage2_TE_mentioned_count_reply_timestamp',
    'stage2_TE_mentioned_count_retweet_timestamp',
    'stage2_TE_mentioned_count_retweet_with_comment_timestamp',
    'stage2_TE_mentioned_count_like_timestamp',
    'stage2_TE_mentioned_bucket_id_reply_timestamp',
    'stage2_TE_mentioned_bucket_id_retweet_timestamp',
    'stage2_TE_mentioned_bucket_id_retweet_with_comment_timestamp',
    'stage2_TE_mentioned_bucket_id_like_timestamp',
    'stage2_GTE_has_mention_engaging_user_id_reply_timestamp',
    'stage2_GTE_has_mention_engaging_user_id_retweet_timestamp',
    'stage2_GTE_has_mention_engaging_user_id_retweet_with_comment_timestamp',
    'stage2_GTE_has_mention_engaging_user_id_like_timestamp',
    'stage2_GTE_mentioned_count_engaging_user_id_reply_timestamp',
    'stage2_GTE_mentioned_count_engaging_user_id_retweet_timestamp',
    'stage2_GTE_mentioned_count_engaging_user_id_retweet_with_comment_timestamp',
    'stage2_GTE_mentioned_count_engaging_user_id_like_timestamp',
    'stage2_GTE_mentioned_bucket_id_engaging_user_id_reply_timestamp',
    'stage2_GTE_mentioned_bucket_id_engaging_user_id_retweet_timestamp',
    'stage2_GTE_mentioned_bucket_id_engaging_user_id_retweet_with_comment_timestamp',
    'stage2_GTE_mentioned_bucket_id_engaging_user_id_like_timestamp',
    'stage2_GTE_language_engaged_with_user_id_reply_timestamp',
    'stage2_GTE_language_engaged_with_user_id_retweet_timestamp',
    'stage2_GTE_language_engaged_with_user_id_retweet_with_comment_timestamp',
    'stage2_GTE_language_engaged_with_user_id_like_timestamp',
    'stage2_GTE_language_engaging_user_id_reply_timestamp',
    'stage2_GTE_language_engaging_user_id_retweet_timestamp',
    'stage2_GTE_language_engaging_user_id_retweet_with_comment_timestamp',
    'stage2_GTE_language_engaging_user_id_like_timestamp',
    'stage2_GTE_dt_dow_engaged_with_user_id_reply_timestamp',
    'stage2_GTE_dt_dow_engaged_with_user_id_retweet_timestamp',
    'stage2_GTE_dt_dow_engaged_with_user_id_retweet_with_comment_timestamp',
    'stage2_GTE_dt_dow_engaged_with_user_id_like_timestamp',
    'stage2_GTE_dt_dow_engaging_user_id_reply_timestamp',
    'stage2_GTE_dt_dow_engaging_user_id_retweet_timestamp',
    'stage2_GTE_dt_dow_engaging_user_id_retweet_with_comment_timestamp',
    'stage2_GTE_dt_dow_engaging_user_id_like_timestamp',
    'stage2_GTE_dt_hour_engaged_with_user_id_reply_timestamp',
    'stage2_GTE_dt_hour_engaged_with_user_id_retweet_timestamp',
    'stage2_GTE_dt_hour_engaged_with_user_id_retweet_with_comment_timestamp',
    'stage2_GTE_dt_hour_engaged_with_user_id_like_timestamp',
    'stage2_GTE_dt_hour_engaging_user_id_reply_timestamp',
    'stage2_GTE_dt_hour_engaging_user_id_retweet_timestamp',
    'stage2_GTE_dt_hour_engaging_user_id_retweet_with_comment_timestamp',
    'stage2_GTE_dt_hour_engaging_user_id_like_timestamp',
    'stage2_GTE_tweet_type_engaged_with_user_id_reply_timestamp',
    'stage2_GTE_tweet_type_engaged_with_user_id_retweet_timestamp',
    'stage2_GTE_tweet_type_engaged_with_user_id_retweet_with_comment_timestamp',
    'stage2_GTE_tweet_type_engaged_with_user_id_like_timestamp',
    'stage2_GTE_tweet_type_engaging_user_id_reply_timestamp',
    'stage2_GTE_tweet_type_engaging_user_id_retweet_timestamp',
    'stage2_GTE_tweet_type_engaging_user_id_retweet_with_comment_timestamp',
    'stage2_GTE_tweet_type_engaging_user_id_like_timestamp',
    'stage2_GTE_engaged_with_user_id_engaging_user_id_reply_timestamp',
    'stage2_GTE_engaged_with_user_id_engaging_user_id_retweet_timestamp',
    'stage2_GTE_engaged_with_user_id_engaging_user_id_retweet_with_comment_timestamp',
    'stage2_GTE_engaged_with_user_id_engaging_user_id_like_timestamp',
    'stage2_GTE_engaged_with_user_id_language_tweet_type_reply_timestamp',
    'stage2_GTE_engaged_with_user_id_language_tweet_type_retweet_timestamp',
    'stage2_GTE_engaged_with_user_id_language_tweet_type_retweet_with_comment_timestamp',
    'stage2_GTE_engaged_with_user_id_language_tweet_type_like_timestamp',
    'stage2_GTE_engaging_user_id_language_tweet_type_reply_timestamp',
    'stage2_GTE_engaging_user_id_language_tweet_type_retweet_timestamp',
    'stage2_GTE_engaging_user_id_language_tweet_type_retweet_with_comment_timestamp',
    'stage2_GTE_engaging_user_id_language_tweet_type_like_timestamp',
    'stage2_TE_engaging_user_id_reply_timestamp',
    'stage2_TE_engaging_user_id_retweet_timestamp',
    'stage2_TE_engaging_user_id_retweet_with_comment_timestamp',
    'stage2_TE_engaging_user_id_like_timestamp',
    'stage2_GTE_language_tweet_type_present_media_reply_timestamp',
    'stage2_GTE_language_tweet_type_present_media_retweet_timestamp',
    'stage2_GTE_language_tweet_type_present_media_retweet_with_comment_timestamp',
    'stage2_GTE_language_tweet_type_present_media_like_timestamp',
    'stage2_TE_present_media_reply_timestamp',
    'stage2_TE_present_media_retweet_timestamp',
    'stage2_TE_present_media_retweet_with_comment_timestamp',
    'stage2_TE_present_media_like_timestamp',
    'stage2_TE_tw_word0_reply_timestamp',
    'stage2_TE_tw_word0_retweet_timestamp',
    'stage2_TE_tw_word0_retweet_with_comment_timestamp',
    'stage2_TE_tw_word0_like_timestamp',
    'stage2_TE_tweet_id_reply_timestamp',
    'stage2_TE_tweet_id_retweet_timestamp',
    'stage2_TE_tweet_id_retweet_with_comment_timestamp',
    'stage2_TE_tweet_id_like_timestamp',
    'stage2_CE_engaged_with_user_id',
    'stage2_CE_engaging_user_id',
    'stage2_CE_language',
    'stage2_CE_present_media',
    'stage2_CE_tw_word0',
    'stage2_CE_tweet_id',
    'stage2_GCE_engaged_with_user_id_language_tweet_type',
    'stage2_GCE_engaging_user_id_language_tweet_type',
    'stage2_GCE_language_tweet_type_present_media',
    'stage2_GCE_engaged_with_user_id_engaging_user_id']

final_feature_list_stage1 = target_list + indexlist + stage1_features_list
final_feature_list_stage2 = final_feature_list_stage1 + stage2_features_list

TE_col_features_stage1 = [
            'engaged_with_user_id',
            'language',
            'dt_dow',
            'tweet_type',
            'most_used_word_bucket_id',
            'second_used_word_bucket_id',
            'mentioned_count',
            'mentioned_bucket_id',
            ['has_mention', 'engaging_user_id'],
            ['mentioned_count', 'engaging_user_id'],
            ['mentioned_bucket_id', 'engaging_user_id'],
            ['language', 'engaged_with_user_id'],
            ['language', 'engaging_user_id'],
            ['dt_dow', 'engaged_with_user_id'],
            ['dt_dow', 'engaging_user_id'],
            ['dt_hour', 'engaged_with_user_id'],
            ['dt_hour', 'engaging_user_id'],
            ['tweet_type', 'engaged_with_user_id'],
            ['tweet_type', 'engaging_user_id'],
            ['engaged_with_user_id','engaging_user_id'],
            ['engaged_with_user_id','language','tweet_type'],
            ['engaging_user_id','language','tweet_type'],
            'engaging_user_id',
            ['language','tweet_type','present_media'],
            'present_media',
            'tw_word0',
            ['engaging_user_is_verified','tweet_type'],
            ['present_domains', 'language', 'engagee_follows_engager', 'tweet_type', 'present_media', 'engaged_with_user_is_verified'],
            ['present_media', 'tweet_type', 'language', 'engaged_with_user_is_verified', 'engaging_user_is_verified', 'engagee_follows_engager'],
            ['tw_original_user0', 'tweet_type', 'language'],
            ['tw_original_user1', 'tweet_type', 'language'],
    ]
TE_col_features_stage1_threshold = {
    "TE_engaged_with_user_id": 0,
    "TE_language": 0,
    "TE_dt_dow": 0,
    "TE_tweet_type": 0,
    "TE_most_used_word_bucket_id": 0,
    "TE_second_used_word_bucket_id": 0,
    "TE_mentioned_count": 0,
    "TE_mentioned_bucket_id": 0,
    "GTE_has_mention_engaging_user_id": 2,
    "GTE_mentioned_count_engaging_user_id": 2,
    "GTE_mentioned_bucket_id_engaging_user_id": 2,
    "GTE_language_engaged_with_user_id": 1,
    "GTE_language_engaging_user_id": 2,
    "GTE_dt_dow_engaged_with_user_id": 2,
    "GTE_dt_dow_engaging_user_id": 4,
    "GTE_dt_hour_engaged_with_user_id": 3,
    "GTE_dt_hour_engaging_user_id": 4,
    "GTE_tweet_type_engaged_with_user_id": 1,
    "GTE_tweet_type_engaging_user_id": 3,
    "GTE_engaged_with_user_id_engaging_user_id": 3,
    "GTE_engaged_with_user_id_language_tweet_type": 1,
    "GTE_engaging_user_id_language_tweet_type": 3,
    "TE_engaging_user_id": 1,
    "GTE_language_tweet_type_present_media": 0,
    "TE_present_media": 0,
    "TE_tw_word0": 0,
    "GTE_engaging_user_is_verified_tweet_type": 0,
    "GTE_present_domains_language_engagee_follows_engager_tweet_type_present_media_engaged_with_user_is_verified": 0,
    "GTE_present_media_tweet_type_language_engaged_with_user_is_verified_engaging_user_is_verified_engagee_follows_engager": 0,
    "GTE_tw_original_user0_tweet_type_language": 0,
    "GTE_tw_original_user1_tweet_type_language": 0}
TE_col_features_stage2 = [
            'engaged_with_user_id',
            'language',
            'dt_dow',
            'tweet_type',
            'most_used_word_bucket_id',
            'second_used_word_bucket_id',
            'mentioned_count',
            'mentioned_bucket_id',
            ['has_mention', 'engaging_user_id'],
            ['mentioned_count', 'engaging_user_id'],
            ['mentioned_bucket_id', 'engaging_user_id'],
            ['language', 'engaged_with_user_id'],
            ['language', 'engaging_user_id'],
            ['dt_dow', 'engaged_with_user_id'],
            ['dt_dow', 'engaging_user_id'],
            ['dt_hour', 'engaged_with_user_id'],
            ['dt_hour', 'engaging_user_id'],
            ['tweet_type', 'engaged_with_user_id'],
            ['tweet_type', 'engaging_user_id'],
            ['engaged_with_user_id','engaging_user_id'],
            ['engaged_with_user_id','language','tweet_type'],
            ['engaging_user_id','language','tweet_type'],
            'engaging_user_id',
            ['language','tweet_type','present_media'],
            'present_media',
            'tw_word0',
            'tweet_id'
    ]
TE_col_excludes = {'dt_dow': ['reply_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']}
CE_col_features = ['engaged_with_user_id', 
                'engaging_user_id',
                'language',
                'present_media',
                'tw_word0',
                'tweet_id',
                ['engaged_with_user_id','language','tweet_type'],
                ['engaging_user_id','language','tweet_type'],
                ['language','tweet_type','present_media'],
                ['engaged_with_user_id','engaging_user_id']
    ]

def extract_rt(x_org):
    x = x_org.lower().replace('[sep]', '').replace('\[cls\] rt @', '@')
    x = x.split('http')[0]
    x = x.rstrip()
    return(x)

def hashit(x):
    uhash = '0' if len(x)<=2 else x
    hash_object = hashlib.md5(uhash.encode('utf-8'))
    return int(hash_object.hexdigest(),16)%2**32

def ret_word( x, rw=0 ):
    x = x.split(' ')
    
    if len(x)>rw:
        return hashit(x[rw])
    elif rw<0:
        if len(x)>0:
            return hashit(x[-1])
        else:
            return 0
    else:
        return 0
    
def extract_hash(text, split_text='@', no=0):
    text = text.lower()
    uhash = ''
    text_split = text.split('@')
    if len(text_split)>(no+1):
        text_split = text_split[no+1].split(' ')
        cl_loop = True
        uhash += clean_text(text_split[0])
        while cl_loop:
            if len(text_split)>1:
                if text_split[1] in ['_']:
                    uhash += clean_text(text_split[1]) + clean_text(text_split[2])
                    text_split = text_split[2:]
                else:
                    cl_loop = False
            else:
                cl_loop = False
                
    return hashit(uhash)

def clean_text(text):
    if len(text)>1:
        if text[-1] in ['!', '?', ':', ';', '.', ',']:
            return(text[:-1])
    return(text)

def check_last_char_quest(x_org):
    if len(x_org)<1:
        return(0)
    x = x_org.replace('[sep]', '')
    x = x.split('http')[0]
    if '#' in x:
        x = x.split('#')[0] + ' '.join(x.split('#')[1].split(' ')[1:])
    if '@' in x:
        x = x.split('@')[0] + ' '.join(x.split('@')[1].split(' ')[1:])
    x = x.rstrip()
    if len(x)<2:
        return(0)
    elif x[-1]=='?' and x[-2]!='!':
        return(1)
    elif x[-1]=='?' and x[-2]=='!':
        return(2)
    elif x[-1]=='!' and x[-2]=='?':
        return(3)
    elif x[-1]=='!' and x[-2]!='?':
        return(4)
    else:
        return(0)


def decodeBertTokenizerAndExtractFeatures(df):
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    
    t1 = timer()
    # fill NaN with ''
    df.fillna({'present_domains':'', 'hashtags':'', 'present_links':'','present_media':'', 'tweet_type':''}, inplace=True)
    
    # generate new features from existing features 
    ## dealing with string feature
    df["tweet"] = df["text_tokens"].apply(lambda x: tokenizer.decode([int(n) for n in x.split('\t')]))
    df["tweet"] = df["tweet"].apply(lambda x: x.replace('https : / / t. co / ', 'https://t.co/').replace('@ ', '@'))
    df["tw_word0"] = df["tweet"].apply(lambda x: ret_word(x,0)).astype('int32') ##??
    df["tw_original_user0"] = df["tweet"].apply(lambda x: extract_hash(x, no=0)) ##??
    df["tw_original_user1"] = df["tweet"].apply(lambda x: extract_hash(x, no=1)) ##??
    df['mention'] = df['tweet'].str.extract(r"[^RT]\s@(\S+)").fillna('')
    df['has_mention'] = df['mention'].apply(lambda x: 0 if x == '' else 1).astype("int32")
    
    # modify target feature values  
    df['reply_timestamp'] = df['reply_timestamp'].apply(lambda x: 1 if x>0 else 0).astype('int32')
    df['retweet_timestamp'] = df['retweet_timestamp'].apply(lambda x: 1 if x>0 else 0).astype('int32')
    df['retweet_with_comment_timestamp'] = df['retweet_with_comment_timestamp'].apply(lambda x: 1 if x>0 else 0).astype('int32')
    df['like_timestamp'] = df['like_timestamp'].apply(lambda x: 1 if x>0 else 0).astype('int32')
    
    # change data type 
    df['engagee_follows_engager'] = df['engagee_follows_engager'].astype("int32")
    
    ## dealing with simple categorical features 
    df['has_photo'] = df['present_media'].str.contains('Photo').replace({False:0, True:1, None:0}).astype("int32")
    df['has_video'] = df['present_media'].str.contains('Vedio').replace({False:0, True:1, None:0}).astype("int32")
    df['has_gif'] = df['present_media'].str.contains('GIF').replace({False:0, True:1, None:0}).astype("int32")
    ## dealing with numeric features 
    df['a_ff_rate'] = df['engaged_with_user_following_count']/df['engaged_with_user_follower_count']
    df['b_ff_rate'] = df['engaging_user_following_count']/df['engaging_user_follower_count']
    df['ab_age_dff'] = df['engaged_with_user_account_creation'] - df['engaging_user_account_creation']
    df['ab_age_rate'] = (df['engaged_with_user_account_creation']+129)/(df['engaging_user_account_creation']+129) 
    df['ab_fing_rate'] = df['engaged_with_user_following_count']/(1+df['engaging_user_following_count'])
    df['ab_fer_rate'] = df['engaged_with_user_follower_count']/(1+df['engaging_user_follower_count'])
    ## dealing with datetime features 
    df['dt_dow'] = pd.to_datetime(df['tweet_timestamp'],unit='s').dt.weekday.apply(lambda x: 1 if x==6 else x+2).astype("int32")
    df['dt_hour'] = pd.to_datetime(df['tweet_timestamp'],unit='s').dt.hour.astype("int32")
    df['dt_minute'] = pd.to_datetime(df['tweet_timestamp'],unit='s').dt.minute.astype("int32")
    ## dealing with simple categorical features 
    df["len_media"] = df["present_media"].apply(lambda x: x.count('\t')+1 if x != '' else 0).astype('int32')
    df['len_domains'] = df['present_domains'].apply(lambda x: 0 if x=='' else len(str(x).split('\t'))).astype("int32")
    df['len_hashtags'] = df['hashtags'].apply(lambda x: 0 if x=='' else len(str(x).split('\t'))).astype("int32")
    df['len_links'] = df['present_links'].apply(lambda x: 0 if x=='' else len(str(x).split('\t'))).astype("int32")
    t2 = timer()
    print("BertTokenizer decode and feature extacting took %.3f" % (t2 - t1))

    return df 


def generate_dict_dfs(df, cols, doSplit, output_name, sep='\t', bucketSize=100):
    
    dict_dfs = []
    
    for i in cols:
        col_name = i
        
        dict_df = pd.DataFrame()
        
        if doSplit:
            dict_df['dict_col'] = df[col_name].apply(lambda x: x.split(sep)).explode('dict_col')
        else:
            dict_df['dict_col'] = df[col_name]
            
        dict_df = dict_df['dict_col'].value_counts().reset_index().rename(columns={'index':'dict_col', 'dict_col':'count'})
        dict_df['bins'] = pd.qcut(dict_df['count'], q=bucketSize, duplicates='drop')
        
        bins = pd.Series(dict_df['bins'].unique())
        labels = bins.sort_values(ascending=True).reset_index().reset_index()
        label_dict = dict(zip(labels[0], labels['level_0']))
        
        dict_df['dict_col_id']= dict_df['bins'].apply(lambda x: label_dict[x])
        dict_df.drop(columns=['bins'], inplace=True)

        dict_df.to_parquet(output_name+f'/{col_name}.parquet')
        dict_dfs.append({'col_name': col_name, 'dict': dict_df})
    
    return dict_dfs


def generate_dictionary(df, cols, doSplit, output_name, sep, bucketSize):    
    dfs = generate_dict_dfs(df, cols, doSplit, output_name, sep, bucketSize)
    return dfs 


def get_col_tgt_src(i):
    if isinstance(i, str):
        col_name = i
        src_name = i
        return (col_name, src_name)
    elif isinstance(i, dict):
        col_name = next(iter(i.keys()))
        src_name = next(iter(i.values()))
        return (col_name, src_name)
    

def find_dict(name, dict_dfs):
    for i in dict_dfs:
        if isinstance(i, tuple):
            for v in i:
                if isinstance(v, str):
                    dict_name = v
                else:
                    dict_df = v
            if str(dict_name) == str(name):
                return dict_df
        else:
            dict_df = i['dict']
            dict_name = i['col_name']
            if str(dict_name) == str(name):
                return dict_df
    return None

    
def index_mapping(x, indexing):
    return [indexing[i] if i in indexing else 0 for i in x]


def categorify(cols, df, dict_dfs, doSplit=False, sep='\t'):
    
    for i in cols:
        col_name, src_name = get_col_tgt_src(i)
        
        dict_df = find_dict(src_name, dict_dfs)
        
        indexing = dict(zip(dict_df.dict_col, dict_df.dict_col_id))
        
        if doSplit:
            df[col_name] = df[src_name].map(lambda x: x.split(sep)).map(lambda x: index_mapping(x, indexing) )
        else:
            df[col_name] = df[src_name].apply(lambda x: indexing[x] if x in indexing else 0)
    
    return df 


def SortIntArrayByFrequency(items):
    
    counted_items = dict(Counter(items))
    
    return  [ k for _ , k in sorted([(v, k) for k, v in counted_items.items()], reverse=True)]


def categorifyFeatures(df, output_name):
    
    # 1. prepare dictionary 
    dict_dfs = []
    
    t1 = timer()
    dict_dfs.extend(generate_dictionary(df, cols=['tweet'], doSplit=True, sep=' ', bucketSize=100, output_name=output_name))
    dict_dfs.extend(generate_dictionary(df, cols=['mention'], doSplit=False, sep='\t', bucketSize=100, output_name=output_name))
    t2 = timer()
    print("Generate Dictionary took %.3f" % (t2 - t1))

    # 2. since we need both mentioned_bucket_id and mentioned_count, add two mention id dict_dfs
    for dict_df in dict_dfs:
        if dict_df['col_name'] == 'mention':
            dict_dfs.append({'col_name': 'mentioned_bucket_id', 'dict':dict_df['dict']})
            dict_dfs.append({'col_name': 'mentioned_count', 'dict':dict_df['dict'].drop(columns='dict_col_id').rename(columns={'count':'dict_col_id'})})
    
    t1 = timer()
    df['mentioned_bucket_id'] = df['mention']
    df['mentioned_count'] = df['mention']
    
    # 3. categorify 
    df = categorify([{'bucketized_tweet_word': 'tweet'}], df, dict_dfs=dict_dfs, doSplit=True, sep=' ')
    df = categorify(['mentioned_bucket_id', 'mentioned_count'], df, dict_dfs=dict_dfs)
    
    # 4. get most and second used bucketized_tweet_word 
    df['sorted_bucketized_tweet_word'] = df['bucketized_tweet_word'].apply(lambda x: SortIntArrayByFrequency(x))
    
    df['most_used_word_bucket_id'] = df['sorted_bucketized_tweet_word'].apply(lambda x: x[0] if len(x)>0 else np.nan)
    df['second_used_word_bucket_id'] = df['sorted_bucketized_tweet_word'].apply(lambda x: x[1] if len(x)>1 else np.nan)                    
    
    t2 = timer()
    print("categorify and getMostAndSecondUsedWordBucketId took %.3f" % (t2 - t1))
    
    return (df, dict_dfs)


class TargetEncoder:
    def __init__(self, inputCols=None, targetCols=None, outputCols=None, target_mean=None, smooth=20, threshold=0):
        self.inputCols = inputCols
        self.targetCols = targetCols
        self.outputCols = outputCols
        self.mean = target_mean
        self.smooth = smooth
        self.threshold = threshold

    def transform(self, dataset, save_path,out_name):
        x_col = self.inputCols
        cols = ['fold', x_col] if isinstance(x_col, str) else ['fold'] + x_col

        aggs = {c:['count','sum'] for c in self.targetCols}
        agg_each_fold = dataset.groupby(cols).agg(aggs)
        agg_each_fold.columns = [col2 + '_' + col1  for col1, col2 in agg_each_fold.columns]
        print("agge each fold is created")
        
        aggs = {c:['count','sum'] for c in self.targetCols}
        agg_all = dataset.groupby(x_col).agg(aggs)
        agg_all.columns = [col2 + '_all_' + col1  for col1, col2 in agg_all.columns]
        print("agge all fold is created")
        
        agg_each_fold = agg_each_fold.join(agg_all, on=x_col, how='right')
        print("agge each fold merge is done")
        
        if self.threshold > 0:
            agg_all = agg_all[agg_all[f'count_all_{self.targetCols[0]}'] > self.threshold]
            
        for i, c in enumerate(self.targetCols):
            out_col = self.outputCols[i]
    
            agg_each_fold[f"count_all_{c}"] = agg_each_fold[f'count_all_{c}'] - agg_each_fold[f'count_{c}']
            agg_each_fold[f"sum_all_{c}"] = agg_each_fold[f'sum_all_{c}'] - agg_each_fold[f'sum_{c}']
            agg_each_fold[out_col] = (agg_each_fold[f"sum_all_{c}"] + self.smooth * self.mean[i]) / (
                        agg_each_fold[f"count_all_{c}"] + self.smooth).astype(float)
            agg_all[out_col] = (agg_all[f'sum_all_{c}'] + self.smooth * self.mean[i]) / (
                        agg_all[f'count_all_{c}'] + self.smooth).astype(float)
            
            agg_each_fold = agg_each_fold.drop(columns=[f"count_all_{c}", f'count_{c}', f"sum_all_{c}",f'sum_{c}'])
            agg_all = agg_all.drop(columns=[f'count_all_{c}', f'sum_all_{c}'])

        agg_each_fold.to_parquet(save_path+'/train' +f'/{out_name}.parquet')
        agg_all.to_parquet(save_path+'/test' +f'/{out_name}.parquet')

        return (agg_each_fold, agg_all)


def TargetEncodingFeatures(df, mode, output_name):   
    if mode == 'stage1':
        features = TE_col_features_stage1
        prefix = ''
    elif mode == 'stage2':
        features = TE_col_features_stage2
        prefix = 'stage2_'
    else:
        raise NotImplementedError("mode need to be train or valid")

    targets = ['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']
    
    t1 = timer()
    y_mean_all = list(df[targets].mean())
    y_mean_all_df = pd.DataFrame(data=[y_mean_all], columns=targets)

    y_mean_all_df.to_parquet(output_name+f'/targets_mean.parquet')

    te_train_dfs = []
    te_test_dfs = []
    for c in features:
        target_tmp = targets
        out_name = ""
        if str(c) in TE_col_excludes:
            target_tmp = []
            for tgt in targets:
                if tgt not in TE_col_excludes[c]:
                    target_tmp.append(tgt)
        out_col_list = []
        for tgt in target_tmp:
            if isinstance(c, list):
                out_col_list.append(prefix + 'GTE_'+'_'.join(c)+'_'+tgt)
                out_name = prefix + 'GTE_'+'_'.join(c)
            else:
                out_col_list.append(prefix + f'TE_{c}_{tgt}')
                out_name = prefix + f'TE_{c}'
        if mode == "stage1":
            threshold = TE_col_features_stage1_threshold[out_name]
        else:
            threshold = 0

        start = timer()
        encoder = TargetEncoder(c, target_tmp, out_col_list, target_mean=y_mean_all,threshold=threshold)
        te_train_df, te_test_df = encoder.transform(df, output_name, out_name)
        te_train_dfs.append({'col_name': ['fold'] + (c if isinstance(c, list) else [c]), 'dict': te_train_df})
        te_test_dfs.append({'col_name': c, 'dict': te_test_df})
        print(f"generating target encoding for %s upon %s took %.1f seconds"%(str(c), str(target_tmp), timer()-start))

    t2 = timer()
    print("Generate encoding feature totally took %.3f" % (t2 - t1))

    return (te_train_dfs, te_test_dfs, y_mean_all_df)

class CountEncoder:
    def __init__(self, proc, x_col_list, y_col_list, out_col_list, out_name, train_generate=True):
        self.op_name = "CountEncoder"
        self.x_col_list = x_col_list
        self.y_col_list = y_col_list
        self.out_col_list = out_col_list
        self.out_name = out_name        
        self.expected_list_size = len(y_col_list)
        self.train_generate = train_generate
      
    def transform(self, df):
        x_col = self.x_col_list
        cols = [x_col] if isinstance(x_col, str) else x_col
        
        agg_all = pd.DataFrame()
        
        for i in range(0, self.expected_list_size):
            y_col = self.y_col_list[i]
            out_col = self.out_col_list[i]
            agg_all = df.groupby([cols]).count()[y_col].reset_index().rename(columns={y_col: out_col})
        
        for i in range(0, self.expected_list_size):
            out_col = self.out_col_list[i]
            agg_all = agg_all[out_col].astype('int32')

        if self.train_generate:
            return (agg_all, agg_all)
        else:
            return agg_all

def CountEncodingFeatures(df, gen_dict, mode, train_generate=True):
    if mode == 'stage1':
        features = CE_col_features
        prefix = ''
    elif mode == 'stage2':
        features = CE_col_features
        prefix = 'stage2_'
    elif mode == 'inference':
        features = CE_col_features
        prefix = "inference_"
    else:
        raise NotImplementedError("mode need to be train or valid")
    
    targets = ['reply_timestamp']

    t1 = timer()
    ce_train_dfs = []
    ce_test_dfs = []
    for c in features:
        target_tmp = targets
        out_name = ""
        out_col_list = []
        for tgt in target_tmp:
            if isinstance(c, list):
                out_col_list.append(prefix + 'GCE_'+'_'.join(c))
                out_name = prefix + 'GCE_'+'_'.join(c)
            else:
                out_col_list.append(prefix + f'CE_{c}')
                out_name = prefix + f'CE_{c}'

        start = timer()
        encoder = CountEncoder(c, target_tmp, out_col_list, out_name,train_generate=train_generate)
        if train_generate:
            ce_train_df, ce_test_df = encoder.transform(df)
            ce_train_dfs.append({'col_name': c, 'dict': ce_train_df})
        else:
            ce_test_df = encoder.transform(df)
        ce_test_dfs.append({'col_name': c, 'dict': ce_test_df})
        print(f"generating count encoding for %s upon %s took %.1f seconds"%(str(c), str(target_tmp), timer()-start))
    
    t2 = timer()
    print("Generate count encoding feature totally took %.3f" % (t2 - t1))

    if train_generate:
        return (ce_train_dfs, ce_test_dfs)
    else:
        return ce_test_dfs


def model_merge(df, te_train_dfs):
    
    for te_train_df in te_train_dfs:
        keys = te_train_df['col_name']
        dict_df = te_train_df['dict']
        df = df.merge(dict_df, on=keys, how='left')
    
    return df 

def create_path(path):
    path = Path(path)

    if path.exists() and path.is_dir():
        shutil.rmtree(path)
    
    os.makedirs(path)
    
    return None


def mergeTargetEncodingFeatures(df, te_train_dfs, output_name, mode):
    
    feature_list = final_feature_list_stage1 if mode == 'stage1' else final_feature_list_stage2
    t1 = timer()
    # merge dicts to original table
    df = model_merge(df, te_train_dfs)
    # select features
    df = df[feature_list]

    df.to_parquet(output_name)
    
    t2 = timer()
    print("Merge Target Encoding Features took %.3f" % (t2 - t1))

    return df

def mergeCountEncodingFeatures(df, ce_train_dfs, output_name):
    # merge dicts to original table
    t1 = timer()
    df = model_merge(df, ce_train_dfs)
    t2 = timer()
    print("Merge Count Encoding Features took %.3f" % (t2 - t1))
    
    return df


def getTargetEncodingFeaturesDicts(get_path, mode, train_dict_load = True):
    if mode == 'stage1':
        features = TE_col_features_stage1
        prefix = ''
    elif mode == 'stage2':
        features = TE_col_features_stage2
        prefix = 'stage2_'
    else:
        raise NotImplementedError("mode need to be stage1 or stage2")

    targets = ['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']
    y_mean_all = []
    y_mean_all_df = pd.read_parquet(
        "%s/targets_mean.parquet" % (get_path))

    te_train_dfs = []
    te_test_dfs = []
    for c in features:
        target_tmp = targets
        out_name = ""
        if str(c) in TE_col_excludes:
            target_tmp = []
            for tgt in targets:
                if tgt not in TE_col_excludes[c]:
                    target_tmp.append(tgt)
        for tgt in target_tmp:
            if isinstance(c, list):
                out_name = prefix + 'GTE_'+'_'.join(c)
            else:
                out_name = prefix + f'TE_{c}'
        if train_dict_load:
            te_train_path = "%s/train/%s" % (get_path, out_name+'.parquet')
            te_train_dfs.append({'col_name': ['fold'] + (c if isinstance(c, list) else [c]), 'dict': pd.read_parquet(te_train_path)})
        
        te_test_path = "%s/test/%s" % (get_path, out_name+'.parquet')
        te_test_dfs.append({'col_name': c, 'dict': pd.read_parquet(te_test_path)})
        
    return (te_train_dfs, te_test_dfs, y_mean_all_df)


def valid_mergeFeatures(df, te_test_dfs, y_mean_all_df, output_name, mode, dict_dfs=None):
    if mode == "stage1":
        # categorify new data with train generated dictionary
        for dict_df in dict_dfs:
            if dict_df['col_name'] == 'mention':
                dict_dfs.append({'col_name': 'mentioned_bucket_id', 'dict':dict_df['dict']})
                dict_dfs.append({'col_name': 'mentioned_count', 'dict':dict_df['dict'].drop(columns='dict_col_id').rename(columns={'count':'dict_col_id'})})
        
        df['mentioned_bucket_id'] = df['mention']
        df['mentioned_count'] = df['mention']
        
        # df = categorify([{'bucketized_tweet_word': 'tweet'}], df, dict_dfs=dict_dfs, doSplit=True, sep=' ')
        # df.to_parquet(output_name+"categorified_bucket")
        # df = categorify(['mentioned_bucket_id', 'mentioned_count'], df, dict_dfs=dict_dfs)
        # df.to_parquet(output_name+"categorified")
        df = pd.read_parquet(output_name+"categorified")
    
        # 4. get most and second used bucketized_tweet_word 
        df['sorted_bucketized_tweet_word'] = df['bucketized_tweet_word'].apply(lambda x: SortIntArrayByFrequency(x))
    
        df['most_used_word_bucket_id'] = df['sorted_bucketized_tweet_word'].apply(lambda x: x[0] if len(x)>0 else np.nan)
        df['second_used_word_bucket_id'] = df['sorted_bucketized_tweet_word'].apply(lambda x: x[1] if len(x)>1 else np.nan)     
    
    # merge target encoding dicts 
    df = model_merge(df, te_test_dfs)
    df.to_parquet(output_name+".full")
        
    te_feature_list = stage1_features_list if mode == "stage1" else stage2_features_list
    for tgt in target_list:
        to_fill_list = []
        for feature in te_feature_list:
            if 'TE_' in feature and tgt in feature:
                to_fill_list.append(feature)
        df[to_fill_list] = df[to_fill_list].fillna(y_mean_all_df.loc[0, tgt])
    
    # select features
    feature_list = final_feature_list_stage1 if mode == "stage1" else final_feature_list_stage2
    df = df[feature_list]

    if mode == "stage1":
        df.to_parquet(output_name)
    elif mode == "stage2":
        df.to_parquet(output_name)
    else:
        raise NotImplementedError("mode need to be stage1 or stage2")


def split_train(df, output_name, sample_ratio=0.083):
    t1 = timer()
    df = df.sample(frac=sample_ratio, random_state=42)
    
    df.to_parquet(output_name)
    t2 = timer()
    print("select train took %.3f seconds" % (t2 - t1))
    return df


def split_valid_byindex(df, train_output, test_output):
    t1 = timer()
    train_df = df[df["is_train"] == 1]
    train_df.to_parquet(train_output)
    t2 = timer()
    print("split to train took %.3f" % (t2 - t1))
    
    t1 = timer()
    test_df = df[df["is_train"] == 0]
    test_df.to_parquet(test_output)
    t2 = timer()
    print("split to test took %.3f" % (t2 - t1))
    
    return (pd.read_parquet(train_output), pd.read_parquet(test_output))


def train():
    
    ############# set up
    path_prefix = ''
    current_path = '/home/vmagent/app/recdp/examples/notebooks/twitter_recsys/datapre_stage1'
    dicts_folder = '/recsys_dicts'

    create_path(path_prefix+current_path)
    create_path(path_prefix+current_path+dicts_folder)
    create_path(path_prefix+current_path+dicts_folder+'/train')
    create_path(path_prefix+current_path+dicts_folder+'/test')

    ############# load data
    t1 = timer()
    df = pd.read_parquet("/home/vmagent/app/recdp/examples/notebooks/twitter_recsys/datapre_stage1_0.03/train1_sample/")
    t2 = timer()
    print("Reading Data took %.3f" % (t2 - t1))

    # def fix_typo(name):
    #     if 'enag' in name:
    #         return name.replace('enag', 'engag')
    #     else:
    #         return name

    #df.rename(columns=fix_typo, inplace=True)
    #df.drop(columns="tokens", inplace=True)
    print('data loaded!')

    ############# decode data
    np.random.seed(42)
    df['fold'] = np.round(np.random.rand(df.shape[0], 1)*4).astype('int32')
    df = decodeBertTokenizerAndExtractFeatures(df)
    df.to_parquet(f"{path_prefix}{current_path}/train_decode")
    print("data decoded!")

    ############# categorify data
    df, dict_dfs = categorifyFeatures(df, output_name=path_prefix+current_path+dicts_folder)
    df.to_parquet(f"{path_prefix}{current_path}/train_categorified")
    print("data categorified!")

    ############# target encoding
    te_train_dfs, te_test_dfs, y_mean_all_df = TargetEncodingFeatures(df, mode="stage1", output_name=path_prefix+current_path+dicts_folder)
    print("data encoded!")

    # ############# select sample data for training
    # print("before select:", df.shape[0])
    # df = split_train(df, output_name= path_prefix+current_path+'/train3_select.parquet', sample_ratio=0.083)
    # print("after select:", df.shape[0])
    # print("data selected!")
    
    ############# merge target encoding only for selected data
    df = mergeTargetEncodingFeatures(df, te_train_dfs, output_name=path_prefix+current_path+"/stage1_train.parquet", mode="stage1")
    print("data merged!")


def valid_stage1():
    ############# set up
    path_prefix = ''
    current_path = '/home/vmagent/app/recdp/examples/notebooks/twitter_recsys/datapre_stage1'
    original_folder = '/home/vmagent/app/recdp/examples/notebooks/twitter_recsys/oridata/valid/valid'
    dicts_folder = '/recsys_dicts'
    index_path = '/home/vmagent/app/recdp/examples/notebooks/twitter_recsys/oridata/valid/valid_split_index.parquet'


    schema = ["text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains", "tweet_type",
            "language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count", "engaged_with_user_following_count",
            "engaged_with_user_is_verified", "engaged_with_user_account_creation", "engaging_user_id", "engaging_user_follower_count",
            "engaging_user_following_count", "engaging_user_is_verified", "engaging_user_account_creation", "engagee_follows_engager",
            "reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"]


    ############# load data
    # df = pd.read_csv(path_prefix+original_folder, sep='\x01', names = schema, header=None)
    # df_index = pd.read_parquet(path_prefix+index_path)
    # df_index = df_index[['tweet_id','engaging_user_id', 'is_train']]
    # df = pd.merge(df, df_index, on=["tweet_id","engaging_user_id"])
    # print("data loaded!")
    
    # ############# decode data
    # df = decodeBertTokenizerAndExtractFeatures(df)
    # print("data decoded!")
    # df.to_parquet(f"{current_path}/stage1_valid_decoded.parquet")
    df = pd.read_parquet(f"{current_path}/stage1_valid_decoded.parquet")

    ############# merge target encoding dict from train
    dict_names = ['tweet', 'mention']
    dict_dfs = [{'col_name': name, 'dict': pd.read_parquet(
            "%s/%s/%s/%s" % (path_prefix, current_path, dicts_folder, name+'.parquet'))} for name in dict_names]
    _, te_test_dfs, y_mean_all_df = getTargetEncodingFeaturesDicts(path_prefix+current_path+dicts_folder, mode='stage1')

    val_df = valid_mergeFeatures(df, te_test_dfs, y_mean_all_df, output_name=path_prefix+current_path+"/stage1_valid.parquet",mode="stage1", dict_dfs=dict_dfs)
    print("val data merged!")


def valid_stage2():
    ############# set up
    path_prefix = '/mnt/data'
    current_path = '/recsys2021/datapre_stage1'
    original_folder = '/valid/valid'
    dicts_folder = '/recsys_dicts'
    index_path = '/valid/valid_split_index.parquet'

    ############# load data
    df = pd.read_parquet(path_prefix+current_path+"stage1_valid_all")
    print("data loaded!")

    ############# count encoding
    ce_train_dfs, ce_test_dfs = CountEncodingFeatures(df, proc, gen_dict=True,mode="stage2")
    print("count encoded!")

    df = mergeCountEncodingFeatures(df, ce_train_dfs, proc, output_name = "valid1_withCE")
    print("count encoding merged!")

    ############# split into train and valid
    print("Before split:", df.count())
    df_train, df_valid = split_valid_byindex(df, train_output="valid2_train", test_output="valid2_valid")
    print("after split, train:", df_train.count())
    print("after split, valid:", df_valid.count())
    print("split done!")

    ############# target encoding of train data
    numFolds = 5
    np.random.seed(42)
    df_train['fold'] = np.round(np.random.rand(df_train.shape[0], 1)*(numFolds-1)).astype('int32')
    te_train_dfs, te_test_dfs, y_mean_all_df = TargetEncodingFeatures(df_train, mode = 'stage2')
    print("target encoded!")

    df_train = mergeTargetEncodingFeatures(df_train, te_train_dfs, output_name='stage2_train', mode='stage2')
    print("train target encoding merged!")

    ############# merge target encoding dict for valid data
    df_valid = valid_mergeFeatures(df_valid, te_test_dfs, y_mean_all_df, output_name="stage2_valid",mode='stage2')
    print("val data merged!")


def inference_decoder():
    ############# set up
    path_prefix = "/mnt/data"
    current_path = "/recsys2021/datapre_stage1"
    original_folder = "/test/test"
    dicts_folder = "/recsys_dicts"

    schema = ["text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains", "tweet_type",
            "language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count", "engaged_with_user_following_count",
            "engaged_with_user_is_verified", "engaged_with_user_account_creation", "engaging_user_id", "engaging_user_follower_count",
            "engaging_user_following_count", "engaging_user_is_verified", "engaging_user_account_creation", "engagee_follows_engager",
            "reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"]
    
    target_cols = ["reply_timestamp","retweet_timestamp","retweet_with_comment_timestamp","like_timestamp"]
    #############  load data
    df = pd.read_csv(path_prefix+original_folder, sep='\x01', names = schema, header=None)
    df[target_cols] = df[target_cols].fillna(0)
    print("data loaded!")

    #############  decode data
    df = decodeBertTokenizerAndExtractFeatures(df)
    df.to_parquet(path_prefix+current_path+'test1_decode.parquet')
    print("data decoded!")


def inference_join():
    ############# set up
    path_prefix = "/mnt/data"
    current_path = "/recsys2021/datapre_stage1"
    original_folder = "/test/test"
    dicts_folder = "/recsys_dicts"
    
    #############  load decoder data
    df = spark.read.parquet(path_prefix+current_path+'test1_decode.parquet')
    print("data decoded!")

    ############# load dict from stage 1
    dict_names = ['tweet', 'mention']
    dict_dfs = [{'col_name': name, 'dict': pd.read_parquet(
            "%s/%s/%s/%s" % (path_prefix, current_path, dicts_folder, name+'.parquet'))} for name in dict_names]
    _, te_test_dfs, y_mean_all_df = getTargetEncodingFeaturesDicts(mode='stage1', train_dict_load=False)
    
    ############# set up to stage 2
    current_path = "/recsys2021/datapre_stage2/"

    ############# count encoding
    ce_test_dfs = CountEncodingFeatures(df, gen_dict=True,mode="inference",train_generate=False)
    
    ############# load dict from stage 2
    _, te_test_dfs_stage2, y_mean_all_df_stage2 = getTargetEncodingFeaturesDicts(get_path, mode = 'stage2', train_dict_load=False)

    ############# final merge
    #df = inference_mergeFeatures(df, dict_dfs, ce_test_dfs,te_test_dfs, te_test_dfs_stage2, y_mean_all_df, y_mean_all_df_stage2, proc, output_name="stage12_test")


if __name__ == "__main__":
    time1 = time.time()
    train_mode = sys.argv[1]
    # cluster_mode = sys.argv[2]

    # if cluster_mode == 'local':
    #     ray.init(runtime_env={'env_vars': {'__MODIN_AUTOIMPORT_PANDAS__': '1'}})
    # else:
    #     ray.init(address="auto", runtime_env={'env_vars': {'__MODIN_AUTOIMPORT_PANDAS__': '1'}}, _temp_dir='/mnt/data/ray_tmp')

    if train_mode == "train":
        train()
    elif train_mode == "valid_stage1":
        valid_stage1()
    elif train_mode == "valid_stage2":
        valid_stage2()
    elif train_mode == "inference_decoder":
        inference_decoder()
    elif train_mode == "inference_join":
        inference_join()

    print(f"Took totally {time.time()-time1} seconds")

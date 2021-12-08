from pyspark.sql.types import *


class RecsysSchema:
    def __init__(self):
        self.string_cols1 = [
            'text_tokens',
            'hashtags',  # Tweet Features
            'tweet_id',       #
            'present_media',          #
            'present_links',          #
            'present_domains',        #
            'tweet_type',     #
            'language',       #
        ]
        self.long_cols1 = [
            'tweet_timestamp',
        ]
        self.string_cols2 = [
            'engaged_with_user_id',
        ]
        self.long_cols2 = [
            'engaged_with_user_follower_count',  # Engaged With User Features
            'engaged_with_user_following_count',      #
        ]
        self.bool_cols1 = [
            'engaged_with_user_is_verified',          #
        ]
        self.long_cols3 = [
            'engaged_with_user_account_creation',
        ]
        self.string_cols3 = [
            'engaging_user_id',
        ]
        self.long_cols4 = [
            'engaging_user_follower_count',  # Engaging User Features
            'engaging_user_following_count',      #
        ]
        self.bool_cols2 = [
            'engaging_user_is_verified',
        ]
        self.long_cols5 = [
            'engaging_user_account_creation',
        ]
        self.bool_cols3 = [
            'engagee_follows_engager',  # Engagement Features

        ]
        self.double_cols = [
            'reply_timestamp',  # Target Reply
            'retweet_timestamp',  # Target Retweet
            'retweet_with_comment_timestamp',  # Target Retweet with comment
            'like_timestamp',  # Target Like
        ]

        # After some conversion
        self.long_cols6 = [
            'tweet_timestamp',
            'engaged_with_user_follower_count',  # Engaged With User Features
            'engaged_with_user_following_count',      #
            'engaged_with_user_account_creation',
            'engaging_user_follower_count',  # Engaging User Features
            'engaging_user_following_count',           #
            'engaging_user_account_creation',
        ]

    def toStructType(self):
        str_fields1 = [StructField('%s' % i, StringType())
                       for i in self.string_cols1]
        long_fields1 = [StructField('%s' % i, LongType())
                       for i in self.long_cols1]
        str_fields2 = [StructField('%s' % i, StringType())
                       for i in self.string_cols2]
        long_fields2 = [StructField('%s' % i, LongType())
                       for i in self.long_cols2]
        bool_fields1 = [StructField('%s' % i, BooleanType())
                        for i in self.bool_cols1]
        long_fields3 = [StructField('%s' % i, LongType())
                       for i in self.long_cols3]
        str_fields3 = [StructField('%s' % i, StringType())
                       for i in self.string_cols3]
        long_fields4 = [StructField('%s' % i, LongType())
                       for i in self.long_cols4]
        bool_fields2 = [StructField('%s' % i, BooleanType())
                        for i in self.bool_cols2]
        long_fields5 = [StructField('%s' % i, LongType())
                       for i in self.long_cols5]
        bool_fields3 = [StructField('%s' % i, BooleanType())
                        for i in self.bool_cols3]
        double_fields = [StructField('%s' % i, DoubleType())
                        for i in self.double_cols]
        return StructType(
            str_fields1
            + long_fields1
            + str_fields2
            + long_fields2
            + bool_fields1
            + long_fields3
            + str_fields3
            + long_fields4
            + bool_fields2
            + long_fields5
            + bool_fields3
            + double_fields
        )

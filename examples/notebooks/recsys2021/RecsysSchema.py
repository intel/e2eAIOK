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
        self.int_cols1 = [
            'tweet_timestamp',
        ]
        self.string_cols2 = [
            'engaged_with_user_id',
        ]
        self.int_cols2 = [
            'engaged_with_user_follower_count',  # Engaged With User Features
            'engaged_with_user_following_count',      #
        ]
        self.bool_cols1 = [
            'engaged_with_user_is_verified',          #
        ]
        self.int_cols3 = [
            'engaged_with_user_account_creation',
        ]
        self.string_cols3 = [
            'engaging_user_id',
        ]
        self.int_cols4 = [
            'enaging_user_follower_count',  # Engaging User Features
            'enaging_user_following_count',      #
        ]
        self.bool_cols2 = [
            'enaging_user_is_verified',
        ]
        self.int_cols5 = [
            'enaging_user_account_creation',
        ]
        self.bool_cols3 = [
            'engagee_follows_engager',  # Engagement Features

        ]
        self.float_cols = [
            'reply_timestamp',  # Target Reply
            'retweet_timestamp',  # Target Retweet
            'retweet_with_comment_timestamp',  # Target Retweet with comment
            'like_timestamp',  # Target Like
        ]

        # After some conversion
        self.int_cols6 = [
            'tweet_timestamp',
            'engaged_with_user_follower_count',  # Engaged With User Features
            'engaged_with_user_following_count',      #
            'engaged_with_user_account_creation',
            'enaging_user_follower_count',  # Engaging User Features
            'enaging_user_following_count',           #
            'enaging_user_account_creation',
        ]

    def toStructType(self):
        str_fields1 = [StructField('%s' % i, StringType())
                       for i in self.string_cols1]
        int_fields1 = [StructField('%s' % i, IntegerType())
                       for i in self.int_cols1]
        str_fields2 = [StructField('%s' % i, StringType())
                       for i in self.string_cols2]
        int_fields2 = [StructField('%s' % i, IntegerType())
                       for i in self.int_cols2]
        bool_fields1 = [StructField('%s' % i, BooleanType())
                        for i in self.bool_cols1]
        int_fields3 = [StructField('%s' % i, IntegerType())
                       for i in self.int_cols3]
        str_fields3 = [StructField('%s' % i, StringType())
                       for i in self.string_cols3]
        int_fields4 = [StructField('%s' % i, IntegerType())
                       for i in self.int_cols4]
        bool_fields2 = [StructField('%s' % i, BooleanType())
                        for i in self.bool_cols2]
        int_fields5 = [StructField('%s' % i, IntegerType())
                       for i in self.int_cols5]
        bool_fields3 = [StructField('%s' % i, BooleanType())
                        for i in self.bool_cols3]
        float_fields = [StructField('%s' % i, FloatType())
                        for i in self.float_cols]
        return StructType(
            str_fields1
            + int_fields1
            + str_fields2
            + int_fields2
            + bool_fields1
            + int_fields3
            + str_fields3
            + int_fields4
            + bool_fields2
            + int_fields5
            + bool_fields3
            + float_fields
        )

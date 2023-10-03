from .base import BaseLLMOperation, LLMOPERATORS
from ray.data import Dataset
import os

def prepare_func_filter_by_length(minimum_length=100, maximum_length=10000):
    def check_length(text):
        if len(text) < minimum_length or (maximum_length != -1 and len(text) > maximum_length):
            return False
        else:
            return True
    return check_length


def prepare_func_filter_by_badwords(language):
    from pyrecdp.primitives.llmutils.utils import get_llmutils_home
    import re
    llmutils_path = get_llmutils_home()
    bad_words_lists_path = os.path.join(llmutils_path, "bad_words_lists", language)
    with open(bad_words_lists_path, "r") as f:
        lines = f.readlines()
    bad_words_list = [s.replace('\n', '') for s in lines]
    bad_words_pattern = " | ".join(bad_words_list)
    def check_badwords(text):
        found = re.search(bad_words_pattern, text)
        if found:
            print(found)
            return False
        else:
            return True
    return check_badwords


def prepare_func_filter_by_profanity():
    from profanity_check import predict
    def check_profanity(text):
        scores = predict([text])
        ret = not bool(scores[0])
        if not ret:
            print(text)
        return ret
    return check_profanity


class LengthFilter(BaseLLMOperation):
    def __init__(self, text_key = 'text', minimum_length=100, maximum_length=-1):
        self.text_key = text_key
        self.inplace = True
        self.minimum_length = minimum_length
        self.maximum_length = maximum_length
        settings = {'text_key': text_key}
        settings.update({'minimum_length': minimum_length, 'maximum_length': maximum_length})
        super().__init__(settings)
        
    def process_rayds(self, ds: Dataset) -> Dataset:
        check_length = prepare_func_filter_by_length(self.minimum_length, self.maximum_length)
        if self.inplace:
            # remove unwanted text row inplace
            return ds.filter(lambda x: check_length(x[self.text_key]))
        else:
            raise NotImplementedError("We only support inplace modification for LengthFilter.")
LLMOPERATORS.register(LengthFilter)


class BadwordsFilter(BaseLLMOperation):
    def __init__(self, text_key = 'text', language = 'en'):
        self.text_key = text_key
        self.inplace = True
        self.language = language
        settings = {'text_key': text_key}
        settings.update({'language': language})
        super().__init__(settings)
        
    def process_rayds(self, ds: Dataset) -> Dataset:
        check_badwords = prepare_func_filter_by_badwords(self.language)
        if self.inplace:
            # remove unwanted text row inplace
            return ds.filter(lambda x: check_badwords(x[self.text_key]))
        else:
            raise NotImplementedError("We only support inplace modification for BadwordsFilter.")

LLMOPERATORS.register(BadwordsFilter)


class ProfanityFilter(BaseLLMOperation):
    def __init__(self, text_key = 'text', inplace = True):
        self.text_key = text_key
        self.inplace = inplace
        settings = {'text_key': text_key}
        super().__init__(settings)
        
    def process_rayds(self, ds: Dataset) -> Dataset:
        check_profanity = prepare_func_filter_by_profanity()
        if self.inplace:
            # remove unwanted text row inplace
            return ds.filter(lambda x: check_profanity(x[self.text_key]))
        else:
            raise NotImplementedError("We only support inplace modification for ProfanityFilter.")
        
LLMOPERATORS.register(ProfanityFilter)


class URLFilter(BaseLLMOperation):
    def __init__(self, text_key = 'text'):
        self.text_key = text_key
        self.inplace = True
        settings = {'text_key': text_key}
        super().__init__(settings)
        
    def process_rayds(self, ds: Dataset) -> Dataset:
        if self.inplace:
            # remove unwanted text row inplace
            pass
        else:
            raise NotImplementedError("We only support inplace modification for URLFilter.")
        
LLMOPERATORS.register(URLFilter)
from pyrecdp.primitives.operations.base import LLMOPERATORS
from pyrecdp.primitives.operations.filter.base import BaseFilter

import os


def prepare_func_filter_by_badwords(language="en"):
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
            return False
        else:
            return True

    return check_badwords


class BadwordsFilter(BaseFilter):
    def __init__(self, language='en'):
        """
            Keeps samples without bad words. The bad words list comes from https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words

            :param language:  Sample in which language. Default: en. Referring https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words to get supported language
        """
        settings = {'language': language}
        super().__init__(args_dict=settings)
        self.language = language

    def get_compute_func(self, *args, **kwargs):
        func = prepare_func_filter_by_badwords(self.language)
        return func


LLMOPERATORS.register(BadwordsFilter)

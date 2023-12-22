"""
 Copyright 2024 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

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

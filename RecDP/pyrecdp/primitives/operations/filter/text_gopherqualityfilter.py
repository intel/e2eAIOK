from pyrecdp.primitives.operations.filter.base import BaseFilter
from pyrecdp.primitives.operations.base import LLMOPERATORS, statistics_decorator
import re
import string

class GopherQualityFilter(BaseFilter):

    def __init__(self, text_key = 'text'):
        settings = {'text_key': text_key}
        super().__init__(settings)
        self.desired_stop_words = set(['the', 'be', 'to', 'of', 'and', 'that', 'have', 'with'])
        self.alphabet = set('abcdefghijklmnopqrstuvwxyz')

    def get_compute_func(self, *args, **kwargs):
        desired_stop_words = self.desired_stop_words
        alphabet = self.alphabet
        def does_word_have_alphabet(word):
            for char in word:
                if char in alphabet:
                    return 1
            return 0
        
        def clean(s):
            s = s.lower()
            s = s.translate(str.maketrans("", "", string.punctuation))
            s = re.sub(r"\s+", " ", s.strip())
            return s

        def compute(text) -> bool:
            stats = {
                'num_words' : 0, 
                'total_word_length' : 0, 
                'num_words_with_alphabet' : 0,
                'num_hash' : 0,
                'num_ellipsis' : 0,
                'num_lines_starting_with_bullet' : 0,
                'num_lines_ending_with_ellipsis' : 0,
                'desired_stop_words_found' : set(),
                'num_lines' : 0
            }
            for line in text.lower().split("."):
                stats['num_lines'] += 1
                words = line.split(" ")
                stats['num_words'] += len(words)
                stats['num_lines_starting_with_bullet'] += int(words[0] == u'\u2022')
                stats['num_lines_ending_with_ellipsis'] += int('...' in words[-1])
                stats['num_hash'] += line.count('#')
                stats['num_ellipsis'] += line.count('...')
                stats['total_word_length'] += sum(len(word) for word in words)
                stats['desired_stop_words_found'].update(set(word for word in clean(line).split(" ") if word in desired_stop_words))
                stats['num_words_with_alphabet'] += sum([does_word_have_alphabet(word) for word in words])
            results = {
                'num_words' : stats['num_words'],
                'mean_word_length' : round(stats['total_word_length'] / stats['num_words'], 2), 
                'num_hash_to_num_words' : round(stats['num_hash'] / stats['num_words'], 2),
                'num_ellipsis_to_num_words' : round(stats['num_ellipsis'] / stats['num_words'], 2),
                'fraction_of_lines_starting_with_bullet' : round(stats['num_lines_starting_with_bullet'] / stats['num_lines'], 2),
                'fraction_of_lines_ending_with_ellipsis' : round(stats['num_lines_ending_with_ellipsis'] / stats['num_lines'], 2),
                'fraction_of_words_with_alphabet' : round(stats['num_words_with_alphabet'] / stats['num_words'], 2),
                'num_desired_stop_words_found' : len(stats['desired_stop_words_found'])
                }
            criterion = [
                results['num_words'] < 50 or results['num_words'] > 100000,
                results['mean_word_length'] < 3 or results['mean_word_length'] > 10,
                results['num_hash_to_num_words'] > 0.1 or results['num_ellipsis_to_num_words'] > 0.1,
                results['fraction_of_lines_starting_with_bullet'] > 0.9,
                results['fraction_of_lines_ending_with_ellipsis'] > 0.3,
                results['fraction_of_words_with_alphabet'] < 0.8,
                results['num_desired_stop_words_found'] < 2
            ]
            prediction = any(criterion)
            return not prediction
        return compute

LLMOPERATORS.register(GopherQualityFilter)

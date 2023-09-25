# Some code here has been modified from:
# https://github.com/togethercomputer/RedPajama-Data/
# --------------------------------------------------------

import regex as re

from ..operator import OPERATORS,Operator


@OPERATORS.register_module('remove_bibliography')
class RemoveBibliography(Operator):
    """Mapper to remove bibliography at the end of documents in Latex
    samples."""

    def __init__(self, *args, **kwargs):
        """
        Initialization method.

        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.pattern = r'(\\appendix|'
        self.pattern += r'\\begin\{references\}|'
        self.pattern += r'\\begin\{REFERENCES\}|'
        self.pattern += r'\\begin\{thebibliography\}|'
        self.pattern += r'\\bibliography\{.*\}'
        self.pattern += r').*$'

    def process(self, sample):
        sample[self.text_key] = re.sub(pattern=self.pattern,
                                       repl=r'',
                                       string=sample[self.text_key],
                                       flags=re.DOTALL)
        return sample

from .base import BaseLLMOperation, LLMOPERATORS
from ray.data import Dataset
from pyspark.sql import DataFrame

import os
import re
from typing import Dict
from selectolax.parser import HTMLParser

CPAT = re.compile("copyright", re.IGNORECASE)
PAT = re.compile("/\\*[^*]*\\*+(?:[^/*][^*]*\\*+)*/")


class SupportedType:
    HTML = 'html'
    LATEX = 'latex'
    CODES = 'codes'


def clean_html(text):
    text = text.replace('<li>', '\n*')
    text = text.replace('</li>', '')
    text = text.replace('<ol>', '\n*')
    text = text.replace('</ol>', '')
    parser = HTMLParser(text)
    return parser.text()


def clean_latex(text):
    non_arg_macros = {}
    non_arg_macros.update(_build_non_arg_macros_dict(text))

    # TODO: macros that take arguments are not supported yet
    arg_macros = {}

    cleaned_text = _clean_text_file(
        file_content=text,
        arg_macros=arg_macros,
        non_arg_macros=non_arg_macros
    )
    return cleaned_text


def clean_codes(text):
    return _clean_copyright_comments(text)


def _clean_text_file(
        file_content: str, arg_macros: Dict, non_arg_macros: Dict
) -> str:
    r""" function takes a tex file as input and returns a cleaned version. The
    cleaned version is a concatenation of the tex files with the
    following modifications:

    - remove all comments (i.e. all lines starting with %)
    - remove everything before the first section-like header
    - remove everything after the first occurrence of either \appendix or
        \bibliography
    - inline-expand definitions and macros

    @param file_content: the content of the tex file as a string.

    @return: cleaned tex file as a string
    """
    # find the first occurence of a \section-like header and replace everything
    # before it with an empty string. This matches the following pattern:
    #   \<section-type>[optional-args]{name}
    pattern = r"^(.*?)("
    pattern += r"\\\bchapter\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
    pattern += r"\\\bpart\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
    pattern += r"\\\bsection\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
    pattern += r"\\\bsubsection\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
    pattern += r"\\\bsubsubsection\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
    pattern += r"\\\bparagraph\b\*?(?:\[(.*?)\])?\{(.*?)\}"
    pattern += r"\\\bsubparagraph\b\*?(?:\[(.*?)\])?\{(.*?)\}"
    pattern += r")"

    # if no section like header is found, then we return an empty string
    if not re.search(pattern, file_content, flags=re.DOTALL):
        return file_content

    # replace everything with the second group of the match (i.e. everything
    # after and including the section header)
    file_content = re.sub(
        pattern=pattern,
        repl=r"\2",
        string=file_content,
        flags=re.DOTALL  # make sure that the dot matches also newlines
    )

    # remove all line comments
    file_content = re.sub(
        pattern=r"(?m)^%.*\n?",
        repl=r"",
        string=file_content,
        flags=re.MULTILINE
    )

    # remove all in comments within a line
    file_content = re.sub(
        # pattern matches a "%" that is not preceded by a backslash (=comment)
        pattern=r"[^\\]%.+$",
        repl=r"",
        string=file_content,
        flags=re.MULTILINE
    )

    # find the first occurence of either \appendix or \bibliography and
    # replace everything after it with an empty string
    pattern = r"("
    pattern += r"\\appendix|"
    pattern += r"\\begin\{references\}|"
    pattern += r"\\begin\{REFERENCES\}|"
    pattern += r"\\begin\{thebibliography\}|"
    pattern += r"\\bibliography\{.*\}"
    pattern += r").*$"

    file_content = re.sub(
        pattern=pattern,
        repl=r'',
        string=file_content,
        flags=re.DOTALL  # make sure that the dot matches also newlines
    )

    # inline-expand all non-arg macros
    for macro_name, macro_value in non_arg_macros.items():
        file_content = re.sub(
            # make pattern grouped to make sure that the macro is not part
            # of a longer alphanumeric word
            pattern=r"(" + macro_name + r")" + r"([^a-zA-Z0-9])",
            # replace the macro with its value and add back the character that
            # was matched after the macro
            repl=macro_value + r"\2",
            string=file_content
        )

    # inline-expand all macros that use args
    # TODO: inline-expand macros with args
    for macro_name, macro_value in arg_macros.items():
        pass

    return file_content


def _build_non_arg_macros_dict(file_content: str) -> Dict[str, str]:
    r""" function takes the content of a tex file and returns a dictionary
    that contains the definitions of all macros that do not use arguments.
    The dictionary is of the form {macro_name: macro_value}.

    @param file_content: the content of the tex file as a string.

    @return: dict
    """
    # regex for extracting \newcommand macros without arguments
    non_arg_nc_reg = re.compile(
        # this regex matches the following:
        # \newcommand{\macro_name}{macro_value}
        # \newcommand*{\macro_name}{macro_value}
        # where macro_name is only allowed to contain letters and numbers;
        # macro_value can contain any character.
        pattern=r'\\\bnewcommand\b\*?\{(\\[a-zA-Z0-9]+?)\}\{(.*?)\}$',
        flags=re.MULTILINE
    )

    # regex for extracting \def macros without arguments
    non_arg_def_reg = re.compile(
        # this regex matches the following:
        # \def\macro_name{macro_value}
        # where macro_name is only allowed to contain letters and numbers;
        # macro_value can contain any character.
        pattern=r'\\def\s*(\\[a-zA-Z0-9]+?)\s*\{(.*?)\}$',
        flags=re.MULTILINE
    )

    # Extract all user-defined LaTeX macros from the preamble
    macros = {}
    for reg in [non_arg_nc_reg, non_arg_def_reg]:
        for match in reg.finditer(file_content):
            # convert the macro name and value to a raw string that can be
            # used in re.sub
            macro_name = match \
                .group(1).encode("unicode-escape").decode("utf-8")
            macro_val = match \
                .group(2).encode("unicode-escape").decode("utf-8")

            macros[macro_name] = macro_val

    return macros


def _clean_copyright_comments(text: str):
    r = PAT.search(text)
    if r:
        # found one, now see if it contains "copyright", if so strip it
        span = r.span()
        sub = text[span[0]:span[1]]
        if CPAT.search(sub):
            # cut it
            text = text[: span[0]] + text[span[1]:]

        return text

    lines = text.split('\n')
    skip = 0

    # Greedy replace any file that begins with comment block, most
    # are copyright headers
    for k in range(len(lines)):
        if (
                lines[k].startswith("//") or
                lines[k].startswith("#") or
                lines[k].startswith("--") or
                not lines[k]
        ):
            skip = skip + 1
        else:
            break

    if skip:
        # we skipped, consume it
        text = "\n".join(lines[skip:])
    return text


def get_fixer_by_type(text_type):
    if isinstance(text_type, str):
        text_type = getattr(SupportedType, text_type.upper())
    if SupportedType.HTML == text_type:
        return clean_html
    if SupportedType.LATEX == text_type:
        return clean_latex
    if SupportedType.CODES == text_type:
        return clean_codes


class TextFix(BaseLLMOperation):
    def __init__(self, text_key='text', inplace=True, text_type='html'):
        """
            Clean up text of the specified type

            :param text_type: Supported text type. Default: html. (html, latex, codes)

        """
        settings = {'text_key': text_key, 'inplace': inplace, 'text_type': text_type}
        super().__init__(settings)
        self.text_key = text_key
        self.inplace = inplace
        self.text_type = text_type
        self.actual_func = None
        self.support_spark = True
        self.support_ray = True

    def process_rayds(self, ds: Dataset) -> Dataset:
        if self.inplace:
            new_name = self.text_key
        else:
            new_name = 'fixed_text'
        if self.actual_func is None:
            self.actual_func = get_fixer_by_type(self.text_type)
        return ds.map(lambda x: self.process_row(x, self.text_key, new_name, self.actual_func))
    
    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        import pyspark.sql.functions as F
        fix_by_type_udf = F.udf(get_fixer_by_type(self.text_type))
        if self.inplace:
            new_name = self.text_key
        else:
            new_name = 'fixed_text'
        return spark_df.withColumn(new_name, fix_by_type_udf(F.col(self.text_key)))


LLMOPERATORS.register(TextFix)

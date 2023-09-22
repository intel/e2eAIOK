import os
import re
from typing import Dict

from selectolax.parser import HTMLParser
from pyspark.sql.types import StructType, StructField, StringType, BooleanType
from pyspark.sql import functions as F

from pyrecdp.core.utils import Timer
from pyrecdp.primitives.llmutils.utils import get_target_file_list, read_json, read_parquet
from pyrecdp.primitives.spark_data_processor.data_processor import DataProcessor as SparkDataProcessor

CPAT = re.compile("copyright", re.IGNORECASE)
PAT = re.compile("/\\*[^*]*\\*+(?:[^/*][^*]*\\*+)*/")


class SupportedType:
    HTML = 'html'
    LATEX = 'latex'
    CODES = 'codes'


@F.udf(returnType=StringType())
def clean_html(text):
    text = text.replace('<li>', '\n*')
    text = text.replace('</li>', '')
    text = text.replace('<ol>', '\n*')
    text = text.replace('</ol>', '')
    parser = HTMLParser(text)
    return parser.text()


@F.udf(returnType=StringType())
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


@F.udf(returnType=StringType())
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


def text_fixer(data_dir, in_type, out_dir, text_types, enable_ray=False):
    if enable_ray:
        rdp = SparkDataProcessor(spark_mode='ray')
    else:
        rdp = SparkDataProcessor()
    spark = rdp.spark
    try:
        with Timer(f"Load data from {in_type} file"):

            data_files = get_target_file_list(data_dir, in_type)
            data_files = [os.path.join(data_dir, f) for f in data_files]
            if in_type == 'parquet':
                spark_df = read_parquet(data_files, spark)
            elif in_type == 'jsonl':
                spark_df = read_json(data_files, spark)
            total_data_num = spark_df.count()

        with Timer("Processing data"):
            fixed_df = text_fixer_spark(spark_df, text_types)

        with Timer("Save data"):
            outfile_path = os.path.join(out_dir, "text_fixed")
            fixed_df.write.mode("overwrite").json(outfile_path)
        print(f"Completed!!")
        print(f"    Loaded and processed total {total_data_num} documents")

    except Exception as e:
        spark.stop()
        print("Failed", e)


def text_fixer_spark(df, text_types):
    fixed_df = df
    for text_type in text_types:
        operator = get_fixer_by_type(text_type)
        fixed_df = fixed_df.withColumn('text', operator(F.col('text')))
    return fixed_df



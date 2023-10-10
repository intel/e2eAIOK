# This tool is referred from alibaba data juicer project and used for
# analyzing the verb-noun structure of the SFT dataset and plots its diversity in sunburst format.

import os

import pandas
import pandas as pd
import spacy
from pyrecdp.core.model_utils import MODEL_ZOO, prepare_model
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import udf

from .base import BaseLLMOperation, LLMOPERATORS
from ray.data import Dataset
from pyspark.sql import DataFrame


# Modify from self_instruct, please refer to
# https://github.com/yizhongw/self-instruct/blob/main/self_instruct/instruction_visualize.ipynb
def find_root_verb_and_its_dobj(tree_root):
    """
    Find the verb and its object closest to the root.

    :param tree_root: the root of lexical tree
    :return: valid verb and its object.
    """
    # first check if the current node and its children satisfy the condition
    if tree_root.pos_ == 'VERB':
        for child in tree_root.children:
            if child.dep_ == 'dobj' and child.pos_ == 'NOUN':
                return tree_root.lemma_ if len(
                    tree_root.lemma_) else tree_root.text, child.lemma_ if len(
                    child.lemma_) else child.text
        return tree_root.lemma_ if len(
            tree_root.lemma_) else tree_root.text, None
    # if not, check its children
    for child in tree_root.children:
        return find_root_verb_and_its_dobj(child)
    # if no children satisfy the condition, return None
    return None, None

# Modify from self_instruct, please refer to
# https://github.com/yizhongw/self-instruct/blob/main/self_instruct/instruction_visualize.ipynb
def find_root_verb_and_its_dobj_in_string(nlp, s, first_sent=True):
    """
    Find the verb and its object closest to the root of lexical tree of input
    string.

    :param nlp: the diversity model to analyse the diversity strings
    :param s: the string to be analysed
    :param first_sent: whether to analyse the first sentence in the
        input string only. If it's true, return the analysis result of
        the first sentence no matter it's valid or not. If it's false,
        return the first valid result over all sentences
    :return: valid verb and its object of this string
    """
    doc = nlp(s)
    for sent in doc.sents:
        verb, noun = find_root_verb_and_its_dobj(sent.root)
        if first_sent or (verb is not None and noun is not None):
            return verb, noun
    return None, None


class DiversityAnalysis:
    """Apply diversity analysis for each sample and get an overall analysis
    result."""

    def __init__(self, dataset, text_key="text", lang_or_model='en'):
        """Initialization method :param dataset: the dataset to be analysed
        :param output_path: path to store the analysis results :param
        lang_or_model: the diversity model or a specific language used to load
        the diversity model."""

        self.dataset = dataset
        self.text_key = text_key
        self.lang_or_model = lang_or_model

    def compute(self, lang_or_model=None):
        """
        Apply lexical tree analysis on each sample.

        :param lang_or_model: the diversity model or a specific language
            used to load the diversity model
        :param column_name: the name of column to be analysed
        :return: the analysis result.
        """

        # load diversity model
        lang_or_model = lang_or_model if lang_or_model else self.lang_or_model
        if isinstance(lang_or_model, str):
            diversity_model = MODEL_ZOO.get(
                prepare_model(lang_or_model, 'spacy'))
        else:
            diversity_model = lang_or_model

        assert isinstance(diversity_model, spacy.Language)

        if isinstance(self.dataset, DataFrame):
            schema = StructType([
                StructField("verb", StringType(), True),
                StructField("noun", StringType(), True)
            ])

            def find_verb_noun_spark(sample):
                try:
                    verb, noun = find_root_verb_and_its_dobj_in_string(
                        diversity_model, sample)
                except Exception as e:
                    print(str(e))
                    verb, noun = None, None
                return verb, noun

            existing_columns = self.dataset.columns
            operator = udf(find_verb_noun_spark, schema)
            dataset = self.dataset.withColumn('diversity', operator(F.col(self.text_key)))
            dataset = dataset.select("diversity.*").toPandas()
        else:
            import copy

            def find_verb_noun_ray(sample):
                try:
                    nlp = copy.deepcopy(diversity_model)
                    verb, noun = find_root_verb_and_its_dobj_in_string(nlp, sample)
                except Exception as e:
                    print(str(e))
                    verb, noun = None, None
                return {'verb': verb, 'noun': noun}
            dataset = self.dataset.map(lambda x: find_verb_noun_ray(x[self.text_key])).to_pandas()
        return dataset

    def get_diversity(self, dataset, top_k_verbs=20, top_k_nouns=4, **kwargs):
        """
        Given the lexical tree analysis result, return the diversity results.

        :param dataset: lexical tree analysis result
        :param top_k_verbs: only keep the top_k_verbs largest verb groups
        :param top_k_nouns: only keep the top_k_nouns largest noun groups
            for each verb group
        :param kwargs: extra args
        :return: the diversity results
        """
        phrases = pd.DataFrame(dataset).dropna()

        top_verbs = phrases.groupby(['verb'
                                     ]).size().nlargest(top_k_verbs).reset_index()

        df = phrases[phrases['verb'].isin(top_verbs['verb'].tolist())]
        df = df.groupby(['verb', 'noun']).size().reset_index().rename(columns={
            0: 'count'
        }).sort_values(by=['count'], ascending=False)

        df = df.groupby('verb').apply(lambda x: x.sort_values(
            'count', ascending=False).head(top_k_nouns)).reset_index(drop=True)
        return df

    def analyse(self,
                lang_or_model=None,
                column_name='text',
                **postproc_kwarg):
        """
        Apply diversity analysis on the whole dataset.

        :param lang_or_model: the diversity model or a specific language
            used to load the diversity model
        :param column_name: the name of column to be analysed
        :param postproc_func: function to analyse diversity. In default,
            it's function get_diversity
        :param postproc_kwarg: arguments of the postproc_func
        :return:
        """
        # get the lexical tree analysis result
        pdf = self.compute(lang_or_model=lang_or_model)
        # get the result of diversity analysis
        df = self.get_diversity(pdf, **postproc_kwarg)
        return df


class TextDiversityIndicate(BaseLLMOperation):
    def __init__(self, text_key='text', language='en', out_dir=''):
        settings = {'text_key': text_key, 'language': language, 'out_dir': out_dir}
        super().__init__(settings)
        self.text_key = text_key
        self.language = language
        self.output_path = out_dir
        self.inplace = False
        self.support_spark = True
        self.support_ray = True

    def process_rayds(self, ds: Dataset) -> Dataset:
        diversity_analysis = DiversityAnalysis(ds, text_key=self.text_key, lang_or_model=self.language)
        analyse_df = diversity_analysis.analyse(lang_or_model=self.language)
        analyse_df.to_csv(os.path.join(self.output_path, 'diversity.csv'))
        analyse_df.to_markdown(os.path.join(self.output_path, 'diversity.md'))
        return ds

    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        diversity_analysis = DiversityAnalysis(spark_df, text_key=self.text_key, lang_or_model=self.language)
        analyse_df = diversity_analysis.analyse(lang_or_model=self.language)
        analyse_df.to_csv(os.path.join(self.output_path, 'diversity.csv'))
        analyse_df.to_markdown(os.path.join(self.output_path, 'diversity.md'))
        return spark_df


LLMOPERATORS.register(TextDiversityIndicate)

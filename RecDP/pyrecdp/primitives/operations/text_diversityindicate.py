# This tool is referred from alibaba data juicer project and used for
# analyzing the verb-noun structure of the SFT dataset and plots its diversity in sunburst format.

import os

import pandas
import pandas as pd
import spacy
from pyrecdp.core.model_utils import MODEL_ZOO, prepare_model
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
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
    verb_noun_pairs = []
    for sent in doc.sents:
        verb, noun = find_root_verb_and_its_dobj(sent.root)
        if first_sent:
            return [(verb, noun)]
        else:
            verb_noun_pairs.append((verb, noun))
    return verb_noun_pairs


class DiversityAnalysis:

    def __init__(self, dataset, text_key="text", lang_or_model='en', first_sent=True):
        """
            Apply diversity analysis for each sample and get an overall analysis result.

            :param dataset: the dataset to be analysed
            :param text_key: the name of column to be analysed
            :param lang_or_model: the diversity model or a specific language used to load the diversity model(spacy model).
            :param first_sent: whether to analyse the first sentence in the input string only.
        """

        self.dataset = dataset
        self.text_key = text_key
        self.lang_or_model = lang_or_model
        self.first_sent = first_sent

    def compute(self, lang_or_model=None):
        """
        Apply lexical tree analysis on each sample.

        :param lang_or_model: the diversity model or a specific language
            used to load the diversity model
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

        first_sent_flag = self.first_sent

        if isinstance(self.dataset, DataFrame):
            schema = ArrayType(
                StructType([
                    StructField("verb", StringType(), True),
                    StructField("noun", StringType(), True),
                ]))

            def find_verb_noun_spark(sample):
                verb_noun_pairs = []
                try:
                    verb_noun_pairs = find_root_verb_and_its_dobj_in_string(
                        diversity_model, sample, first_sent=first_sent_flag)
                except Exception as e:
                    print(str(e))
                return verb_noun_pairs


            operator = udf(find_verb_noun_spark, schema)
            spark_df = self.dataset.withColumn('diversity', operator(F.col(self.text_key)))
            pd_df = spark_df.select("diversity").toPandas()
            df_explode = pd_df.explode('diversity').reset_index(drop=True)
            df_explode = df_explode.dropna()
            dataset = df_explode.apply(lambda x: [x['diversity']['verb'], x['diversity']['noun']], axis=1,
                                       result_type='expand')
            dataset.columns = ['verb', 'noun']
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
    def __init__(self, text_key='text', language='en', out_dir='', first_sent=True):
        settings = {'text_key': text_key, 'language': language, 'out_dir': out_dir, 'first_sent': first_sent}
        super().__init__(settings)
        self.text_key = text_key
        self.language = language
        self.output_path = out_dir
        self.inplace = False
        self.support_spark = True
        self.support_ray = False
        self.first_sent = first_sent

    def process_rayds(self, ds: Dataset) -> Dataset:
        diversity_analysis = DiversityAnalysis(ds, text_key=self.text_key, lang_or_model=self.language,
                                               first_sent=self.first_sent)
        analyse_df = diversity_analysis.analyse(lang_or_model=self.language)
        analyse_df.to_csv(os.path.join(self.output_path, 'diversity.csv'))
        analyse_df.to_markdown(os.path.join(self.output_path, 'diversity.md'))
        return ds

    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        diversity_analysis = DiversityAnalysis(spark_df, text_key=self.text_key, lang_or_model=self.language,
                                               first_sent=self.first_sent)
        analyse_df = diversity_analysis.analyse(lang_or_model=self.language)
        analyse_df.to_csv(os.path.join(self.output_path, 'diversity.csv'))
        analyse_df.to_markdown(os.path.join(self.output_path, 'diversity.md'))
        return spark_df


LLMOPERATORS.register(TextDiversityIndicate)

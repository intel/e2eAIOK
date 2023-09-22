# This tool is referred from alibaba data juicer project and used for
# predicting a document score for text samples using
# quality classifier models we provided, including:
#   - gpt3: A GPT3 quality classifier reproduced from scratch by us based on
#       PySpark. It's trained over CC as negative samples and Wikipedia-en,
#       Books, OpenWebText as positive samples.
#   - chinese: A quality classifier for Chinese. It's trained over Chinese
#       texts sampled from CC as negative samples and Wudao, Wikipedia-zh as
#       positive samples.
#   - code: A quality classifier for codes. It's trained over code samples that
#       have stars >= 1372 as positive samples and random samples from left
#       data as negative samples. Stars count 1372 splits a nearly 700w subset
#       with most stars.
# All these 3 classifiers are trained using the same training pipeline as GPT3
# based on PySpark but with different tokenizers and keeping methods:
#   - gpt3: standard Tokenizer from spark & GPT3 keeping method based on pareto
#   - chinese: sentencepiece tokenizer for Chinese & label
#   - code: sentencepiece tokenizer for code & label
#
# This tool needs several arguments:
#   - dataset_path: the path to the dataset you want to predict doc_scores for.
#   - result_path: the path to store the predicted result dataset.
#   - model: quality classifier name to apply. It's "gpt3" in default. You can
#       use one of ["gpt3", "chinese", "code"] we provided, or you can set it
#       to the path to your own model trained using the train.py tool.
#   - tokenizer: what tokenizer to use to tokenize texts. It's None in default,
#       which means using the standard Tokenizer of PySpark. You can use one of
#       ["zh.sp.model", "code.sp.model"] we provided, or you can set it to the
#       path to your own sentencepiece model.
#   - keep_method: the method to label should_keep field for each sample. It's
#       "gpt3" in default. Should be one of ["gpt3", "label"].
#   - text_key: the field key name to hold texts to be classified. It's "text"
#       in default.
#   - overall_stats: whether to output an overall stats report on predicted
#       document scores. It's False in default.
#
# Recommended arguments for provided trained models:
#   - gpt3:
#       - model: gpt3
#       - tokenizer: None
#       - keep_method: gpt3
#   - chinese:
#       - model: chinese
#       - tokenizer: zh.sp.model
#       - keep_method: label
#   - code:
#       - model: code
#       - tokenizer: code.sp.model
#       - keep_method: label
#
# Notice:
#   1. The configs of SparkSession in function init_spark can be modified to be
#       more suitable for your own machine. See function init_spark in
#       qc_utils.py.
#   2. Random factors are involved in "gpt3" model. So you might get different
#       should_keep label in different running processes. But you should get
#       same doc_score predictions in different running processes.

import argparse
import wget
import zipfile

from pyrecdp.core.utils import Timer
from pyrecdp.primitives.spark_data_processor.data_processor import DataProcessor as SparkDataProcessor
from pyrecdp.primitives.llmutils.utils import *

import numpy as np
import sentencepiece as spm

from loguru import logger
from pyspark.ml import PipelineModel

from pyspark.sql.functions import col, rand, udf
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StringType


DEFAULT_CACHE_HOME = '~/.cache'
CACHE_HOME = os.getenv('CACHE_HOME', DEFAULT_CACHE_HOME)

# Default recdp cache location
DEFAULT_RECDP_CACHE_HOME = os.path.join(CACHE_HOME, 'recdp')
RECDP_CACHE_HOME = os.path.expanduser(
    os.getenv('RECDP_CACHE_HOME', DEFAULT_RECDP_CACHE_HOME))

# Default assets cache location
DEFAULT_RECDP_ASSETS_CACHE = os.path.join(RECDP_CACHE_HOME, 'assets')
RECDP_ASSETS_CACHE = os.getenv('RECDP_ASSETS_CACHE', DEFAULT_RECDP_ASSETS_CACHE)
# Default models cache location
DEFAULT_RECDP_MODELS_CACHE = os.path.join(RECDP_CACHE_HOME, 'models')
RECDP_MODELS_CACHE = os.getenv('RECDP_MODELS_CACHE', DEFAULT_RECDP_MODELS_CACHE)

MODEL_LINKS = 'https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/data_juicer/models/'
BACKUP_MODEL_LINKS = {
    # tokenizer and language model for English from sentencepiece and KenLM
    '%s.sp.model':
    'https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/',
    '%s.arpa.bin':
    'https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/',

    # sentence split model from nltk punkt
    'punkt.%s.pickle':
    'https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/'
    'data_juicer/models/'
}

def prepare_model(model_name, model_path=RECDP_MODELS_CACHE):
    print(f"model_name is {model_name}")

    udm = False
    if model_name not in ['gpt3', 'chinese', 'code']:
        # use user-specific mdoel
        real_model_path = model_name
        udm = True
    else:
        # use prepared models we provided
        model_name = '%s_quality_model' % model_name
        real_model_path = os.path.join(model_path, model_name)
    logger.info(f'Preparing scorer model in [{real_model_path}]...')
    if os.path.exists(real_model_path) and os.path.isdir(real_model_path):
        print(f"real_model_path is {real_model_path}")
        return PipelineModel.load(f"file://{real_model_path}")
    if udm:
        logger.error(f'Customized model [{real_model_path}] cannot be loaded.')
        exit(0)
    # No specific models in local file systems. Download them from remote.
    os.makedirs(model_path, exist_ok=True)
    wget.download(os.path.join(MODEL_LINKS, f'{model_name}.zip'),
                  os.path.join(model_path, f'{model_name}.zip'),
                  bar=None)
    with zipfile.ZipFile(os.path.join(model_path, f'{model_name}.zip')) as zip:
        zip.extractall(os.path.join(model_path))
    return PipelineModel.load(f"file://{real_model_path}")


def load_dataset(spark, ds_path, text_key='text', only_text=False, file_system_prefix=''):
    """
    Load a single dataset using PySpark. Only support 'json', 'jsonl', or
    'parquet' files for now
    :param spark: spark session
    :param ds_path: dataset path
    :param text_key: the name of the column that stores the contents of texts
    :param only_text: whether to load texts only and drop other columns.
    :return: a data frame
    """
    # load dataset using different methods according to the suffix
    ds_path = f"{file_system_prefix}{ds_path}"
    logger.info(f'Loading dataset from [{ds_path}]...')
    if ds_path.endswith('.json') or ds_path.endswith('.jsonl'):
        df = spark.read.json(ds_path)
    elif ds_path.endswith('.parquet'):
        df = spark.read.parquet(ds_path)
    else:
        raise NotImplementedError('Dataset type is not supported for now. '
                                  'Suffix of dataset file should be one of '
                                  '[.json, .jsonl, .parquet]')
    # rename the column that stores texts to "text" if necessary
    if text_key != 'text':
        df = df.withColumnRenamed(text_key, 'text')
    # whether to keep "text" column only
    if only_text:
        return df.select('text')
    else:
        return df


def check_model(model_name, args=(), force=False):
    """
    Check whether a model exists in MODEL_PATH. If exists, return its full path
    Else, download it from cached models links.

    :param model_name: a specified model name
    :param args: optional extra args of model.
    :param force: Whether to download model forcefully or not, Sometimes
        the model file maybe incomplete for some reason, so need to
        download again forcefully.
    """
    if not os.path.exists(RECDP_MODELS_CACHE):
        os.makedirs(RECDP_MODELS_CACHE)

    # check if the specified model exists. If it does not exist, download it
    true_model_name = model_name % args
    mdp = os.path.join(RECDP_MODELS_CACHE, true_model_name)
    if force:
        if os.path.exists(mdp):
            os.remove(mdp)
            logger.info(
                f'Model [{true_model_name}] invalid, force to downloading...')
        else:
            logger.info(
                f'Model [{true_model_name}] not found . Downloading...')

        try:
            model_link = os.path.join(MODEL_LINKS, true_model_name)
            wget.download(model_link, mdp, bar=None)
        except:  # noqa: E722
            try:
                backup_model_link = os.path.join(
                    BACKUP_MODEL_LINKS[model_name], true_model_name)
                wget.download(backup_model_link, mdp, bar=None)
            except:  # noqa: E722
                logger.error(
                    f'Downloading model [{true_model_name}] error. '
                    f'Please retry later or download it into {RECDP_MODELS_CACHE} '
                    f'manually from {model_link} or {backup_model_link} ')
                exit(1)
    return mdp


def prepare_sentencepiece_model(model_name, lang):
    """
    Prepare and load a sentencepiece model.

    :param model_name: input model name in formatting syntax
    :param lang: language to render model name
    :return: model instance.
    """
    import sentencepiece
    logger.info('Loading sentencepiece model...')
    sentencepiece_model = sentencepiece.SentencePieceProcessor()
    try:
        sentencepiece_model.load(check_model(model_name, lang))
    except:  # noqa: E722
        sentencepiece_model.load(check_model(model_name, lang, force=True))
    return sentencepiece_model


def tokenize_dataset(ds, tokenizer):
    """
    Tokenize the texts in input dataset using specified tokenizer
    :param ds: dataset to be tokenized
    :param tokenizer: tokenizer used to tokenize texts
    :return: a dataset with an extra column "words" that stores the tokenized
        texts
    """
    if os.path.exists(tokenizer):
        # if it's a local model
        tkn = spm.SentencePieceProcessor()
        tkn.load(tokenizer)
    else:
        # else, try to load it from our remote model list
        tkn = prepare_sentencepiece_model(tokenizer, ())
    # create a PySpark udf to tokenize the dataset
    tokenizer_udf = udf(lambda text: tkn.encode_as_pieces(text),
                        ArrayType(StringType()))
    logger.info('Tokenize texts using specific tokenizer...')
    return ds.withColumn('words', tokenizer_udf(col('text')))


def get_keep_method_udf(keep_method):
    """
    Given the name of keep method, return a PySpark user-defined function of
    this kind of keep method. Only support 'gpt3' or 'label' for now
    :param keep_method: name of keep method
    :return: a PySpark udf of specified keep method
    """
    if keep_method == 'label':
        return udf(lambda score: int(score > 0.5), IntegerType())
    elif keep_method == 'gpt3':
        pareto = 9
        return udf(lambda score: int(score > 1 - np.random.pareto(pareto)),
                   IntegerType())
    else:
        raise NotImplementedError(f'Keep method [{keep_method}] is not '
                                  f'implemented for now.')


def predict(model, ds, tokenizer=None, keep_method='label'):
    """
    Predict document scores for a dataset using a trained quality classifier
    model
    :param model: the model used to predict
    :param ds: the dataset to be predicted
    :param tokenizer: specified sentencepiece tokenizer. It's None in default,
        which means using the standard Tokenizer in PySpark
    :param keep_method: name of keep method to label the "should_keep" column
    :return:
    """
    logger.info('Start scoring dataset...')
    if tokenizer:
        # tokenizer is not standard Tokenizer in PySpark, need to apply it
        # explicitly
        ds = tokenize_dataset(ds, tokenizer)

    prediction = model.transform(ds)

    # A UDF to extract doc scores from probability vectors
    def extract_prob(v):
        try:
            return float(v[1])
        except ValueError:
            return None

    # extract the predicted probability as the doc_score
    extract_prob_udf = udf(extract_prob, DoubleType())
    doc_score = prediction.withColumn('doc_score',
                                      extract_prob_udf(col('probability')))

    # A UDF to get the bool value indicating whether this sample should be kept
    should_keep_label_udf = get_keep_method_udf(keep_method)
    should_keep = doc_score.withColumn('should_keep',
                                       should_keep_label_udf(col('doc_score')))
    # drop extra useless columns
    return should_keep.drop('words', 'features', 'rawPrediction',
                            'probability', 'prediction')


def export_result(ds, res_path, file_system_prefix=''):
    res_path = f"{file_system_prefix}{res_path}"
    logger.info(f'Exporting predicted result to [{res_path}]')
    if res_path.endswith('.json') or res_path.endswith('.jsonl'):
        ds.write.mode('overwrite').format('json').save(res_path)
    elif res_path.endswith('.parquet'):
        ds.write.mode('overwrite').format('parquet').save(res_path)
    else:
        ds.write.mode('overwrite').save(res_path)


def quality_classifier_spark(df_spark,
                       model='gpt3',
                       tokenizer=None,
                       keep_method='gpt3',
                       text_key='text'):
    """
    Use specific quality classifier to predict document scores on your dataset
    :param df_spark: spark dataframe
    :param model: quality classifier name to apply. It's "gpt3" in default. You
        can use one of ["gpt3", "chinese", "code"] we provided, or you can set
        it to the path to your own model trained using the train.py tool
    :param tokenizer: what tokenizer to use to tokenize texts. It's None in
        default, which means using the standard Tokenizer of PySpark. You can
        use one of ["zh.sp.model", "code.sp.model"] we provided, or you can set
        it to the path to your own sentencepiece model
    :param keep_method: the method to label should_keep field for each sample.
        It's "gpt3" in default. Should be one of ["gpt3", "label"]
    :param text_key: the field key name to hold texts to be classified. It's
        "text" in default
    :return:
    """
    # set default tokenizers for default models
    if model == 'chinese':
        tokenizer = 'zh.sp.model'
        keep_method = 'label'
    if model == 'code':
        tokenizer = 'code.sp.model'
        keep_method = 'label'
    if model == 'gpt3':
        tokenizer = None
        keep_method = 'gpt3'

    # load the quality classifier model
    model = prepare_model(model_name=model)
    # rename predicted column
    if text_key != 'text':
        df_spark = df_spark.withColumnRenamed(text_key, 'text')
    # start to predict
    pred = predict(model, df_spark, tokenizer=tokenizer, keep_method=keep_method).cache()
    return pred


def quality_classifier(dataset_path,
                       result_path,
                       model='gpt3',
                       tokenizer=None,
                       keep_method='gpt3',
                       text_key='text',
                       overall_stats=False,
                       file_system_prefix=''):
    """
    Use specific quality classifier to predict document scores on your dataset
    :param dataset_path: the path to the dataset you want to predict for
    :param result_path: the path to store the predicted result dataset
    :param model: quality classifier name to apply. It's "gpt3" in default. You
        can use one of ["gpt3", "chinese", "code"] we provided, or you can set
        it to the path to your own model trained using the train.py tool
    :param tokenizer: what tokenizer to use to tokenize texts. It's None in
        default, which means using the standard Tokenizer of PySpark. You can
        use one of ["zh.sp.model", "code.sp.model"] we provided, or you can set
        it to the path to your own sentencepiece model
    :param keep_method: the method to label should_keep field for each sample.
        It's "gpt3" in default. Should be one of ["gpt3", "label"]
    :param text_key: the field key name to hold texts to be classified. It's
        "text" in default
    :param overall_stats: whether to output an overall stats report on
        predicted document scores. It's False in default
    :param file_system_prefix: support local or hdfs file system (hdfs:// or file://)
    :return:
    """
    # set default tokenizers for default models
    if model == 'chinese':
        tokenizer = 'zh.sp.model'
        keep_method = 'label'
    if model == 'code':
        tokenizer = 'code.sp.model'
        keep_method = 'label'
    if model == 'gpt3':
        tokenizer = None
        keep_method = 'gpt3'

    # initialize a spark session
    rdp = SparkDataProcessor()
    spark = rdp.spark
    # load the quality classifier model
    model = prepare_model(model_name=model)
    # load dataset
    ds = load_dataset(spark, dataset_path, text_key=text_key, file_system_prefix=file_system_prefix)
    # start to predict
    pred = predict(model, ds, tokenizer=tokenizer, keep_method=keep_method)
    # export prediction result to specific path
    export_result(pred, result_path, file_system_prefix)

    # generate overall statistics on doc scores
    if overall_stats:
        overall = pred.select('doc_score').toPandas().describe(include='all')
        # export to result report file
        overall.to_csv(os.path.join(result_path, 'overall.csv'))
        overall.to_markdown(os.path.join(result_path, 'overall.md'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", dest="dataset_path", type=str)
    parser.add_argument("--result_path", dest="result_path", type=str)
    parser.add_argument("--model", dest="model", type=str, default="gpt3")
    parser.add_argument("--tokenizer", dest="tokenizer", type=str, default=None)
    parser.add_argument("--keep_method", dest="keep_method", type=str, default="gpt3")
    parser.add_argument("--text_key", dest="text_key", type=str, default="text")
    parser.add_argument("--overall_stats", dest="overall_stats", type=bool, default=False)
    parser.add_argument("--file_system_prefix", dest="file_system_prefix", type=str, default="")
    parser.add_argument("--file_type", dest="file_type", type=str, default="jsonl")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    result_path = args.result_path
    model = args.model
    tokenizer = args.tokenizer
    keep_method = args.keep_method
    text_key = args.text_key
    overall_stats = args.overall_stats
    file_system_prefix = args.file_system_prefix
    file_type = args.file_type

    data_files = get_target_file_list(dataset_path, "jsonl", file_system_prefix)

    with Timer(f"Generate language_identify data for {dataset_path}"):
        for file in data_files:
            file_path = os.path.join(dataset_path, file)
            save_path = os.path.join(result_path, file)
            quality_classifier(file_path, save_path, model, tokenizer, keep_method, text_key, overall_stats, file_system_prefix)

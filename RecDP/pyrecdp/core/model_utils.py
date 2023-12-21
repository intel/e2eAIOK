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

""" This code is adapted from Alibaba data-juicer
https://github.com/alibaba/data-juicer/blob/main/data_juicer/utils/model_utils.py
"""
import os
from typing import Callable, Any

from loguru import logger

from .cache_utils import RECDP_MODELS_CACHE
from .import_utils import check_availability_and_install

# Default directory to store models
MODEL_PATH = RECDP_MODELS_CACHE

# Default backup cached models links for downloading
BACKUP_MODEL_LINKS = {
    # language identification model from fasttext
    'lid.176.bin':
        'https://dl.fbaipublicfiles.com/fasttext/supervised-models/',

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

# Default cached models links for downloading
MODEL_LINKS = 'https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/' \
              'data_juicer/models/'

MODEL_ZOO = {}


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
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    # check if the specified model exists. If it does not exist, download it
    true_model_name = model_name % args
    mdp = os.path.join(MODEL_PATH, true_model_name)
    if os.path.exists(mdp):
        if force:
            os.remove(mdp)
            logger.info(
                f'Model [{true_model_name}] removed, force to downloading...')
            download_nltk_model(mdp, model_name, true_model_name)
        else:
            return mdp
    else:
        logger.info(
            f'Model [{true_model_name}] not found . Downloading...')
        download_nltk_model(mdp, model_name, true_model_name)
    return mdp


def download_nltk_model(mdp, model_name, true_model_name):
    import wget
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
                f'Please retry later or download it into {MODEL_PATH} '
                f'manually from {model_link} or {backup_model_link} ')
            exit(1)


def prepare_fasttext_model(model_name):
    """
    Prepare and load a fasttext model.

    :param model_name: input model name
    :return: model instance.
    """
    check_availability_and_install("fasttext==0.9.2")
    import fasttext
    logger.info('Loading fasttext language identification model...')
    try:
        ft_model = fasttext.load_model(check_model(model_name))
    except:  # noqa: E722
        ft_model = fasttext.load_model(check_model(model_name, force=True))
    return ft_model


def prepare_sentencepiece_model(model_name, lang):
    """
    Prepare and load a sentencepiece model.

    :param model_name: input model name in formatting syntax
    :param lang: language to render model name
    :return: model instance.
    """
    check_availability_and_install("sentencepiece")
    import sentencepiece
    logger.info('Loading sentencepiece model...')
    sentencepiece_model = sentencepiece.SentencePieceProcessor()
    try:
        sentencepiece_model.load(check_model(model_name, lang))
    except:  # noqa: E722
        sentencepiece_model.load(check_model(model_name, lang, force=True))
    return sentencepiece_model


def prepare_kenlm_model(model_name, lang):
    """
    Prepare and load a kenlm model.

    :param model_name: input model name in formatting syntax.
    :param lang: language to render model name
    :return: model instance.
    """
    check_availability_and_install("kenlm")
    import kenlm
    logger.info('Loading kenlm language model...')
    try:
        kenlm_model = kenlm.Model(check_model(model_name, lang))
    except:  # noqa: E722
        kenlm_model = kenlm.Model(check_model(model_name, lang, force=True))
    return kenlm_model


def prepare_nltk_model(model_name, lang):
    """
    Prepare and load a nltk punkt model.

    :param model_name: input model name in formatting syntax
    :param lang: language to render model name
    :return: model instance.
    """
    check_availability_and_install("nltk")
    nltk_to_punkt = {
        'en': 'english',
        'fr': 'french',
        'pt': 'portuguese',
        'es': 'spanish'
    }
    assert lang in nltk_to_punkt.keys(
    ), 'lang must be one of the following: {}'.format(
        list(nltk_to_punkt.keys()))

    from nltk.data import load
    logger.info('Loading nltk punkt split model...')
    try:
        nltk_model = load(check_model(model_name, nltk_to_punkt[lang]))
    except:  # noqa: E722
        nltk_model = load(
            check_model(model_name, nltk_to_punkt[lang], force=True))
    return nltk_model


def prepare_huggingface_tokenizer(tokenizer_name):
    """
    Prepare and load a tokenizer from HuggingFace.

    :param tokenizer_name: input tokenizer name
    :return: a tokenizer instance.
    """
    from transformers import AutoTokenizer
    logger.info('Loading tokenizer from HuggingFace...')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                              trust_remote_code=True)
    return tokenizer


def prepare_diversity_model(model_name, lang):
    """
    Prepare diversity model for specific language.

    :param model_name: the model name to be loaded.
    :param lang: language of diversity model. Should be one of ["zh",
        "en"]
    :return: corresponding diversity model
    """
    assert lang in ['zh', 'en'], 'Diversity only support zh and en'
    model_name = model_name % lang
    logger.info(f'Loading spacy model [{model_name}]...')
    compressed_model = '%s.zip' % model_name

    # decompress the compressed model if it's not decompressed
    def decompress_model(compressed_model_path):
        decompressed_model_path = compressed_model_path.replace('.zip', '')
        if os.path.exists(decompressed_model_path) \
                and os.path.isdir(decompressed_model_path):
            return decompressed_model_path
        import zipfile
        with zipfile.ZipFile(compressed_model_path) as zf:
            zf.extractall(MODEL_PATH)
        return decompressed_model_path

    try:
        import spacy
        diversity_model = spacy.load(
            decompress_model(check_model(compressed_model)))
    except:  # noqa: E722
        import spacy
        diversity_model = spacy.load(
            decompress_model(check_model(compressed_model, force=True)))
    return diversity_model


def prepare_model(lang='en', model_type='sentencepiece', model_key=None,
                  prepare_model_func: Callable[[str, str], Any] = None):
    """
    Prepare and load a model or a tokenizer from MODEL_ZOO.
    :param prepare_model_func: a Callable function to prepare model
    :param lang: which lang model to load
    :param model_type: model or tokenizer type
    :param model_key: tokenizer name, only used when prepare HuggingFace
        tokenizer
    :return: a model or tokenizer instance
    """

    type_to_name = {
        'fasttext': ('lid.176.bin', prepare_fasttext_model),
        'sentencepiece': ('%s.sp.model', prepare_sentencepiece_model),
        'kenlm': ('%s.arpa.bin', prepare_kenlm_model),
        'nltk': ('punkt.%s.pickle', prepare_nltk_model),
        'huggingface': ('%s', prepare_huggingface_tokenizer),
        'spacy': ('%s_core_web_md-3.5.0', prepare_diversity_model),
    }
    assert model_type in type_to_name.keys(
    ), 'model_type must be one of the following: {}'.format(
        list(type_to_name.keys()))

    if model_key is None:
        model_key = model_type + '_' + lang
    if model_key not in MODEL_ZOO.keys():
        model_name, model_func = type_to_name[model_type]
        if model_type == 'fasttext':
            MODEL_ZOO[model_key] = model_func(model_name)
        elif model_type == 'huggingface':
            MODEL_ZOO[model_key] = model_func(model_key)
        else:
            if prepare_model_func is not None:
                MODEL_ZOO[model_key] = prepare_model_func(model_name, lang)
            else:
                MODEL_ZOO[model_key] = model_func(model_name, lang)
    return model_key


def get_model(model_key, lang='en', model_type='sentencepiece'):
    """
    Get a model or a tokenizer from MODEL_ZOO.

    :param model_key: name of the model or tokenzier
    """
    if model_key not in MODEL_ZOO:
        prepare_model(lang=lang, model_type=model_type, model_key=model_key)
    return MODEL_ZOO.get(model_key, None)

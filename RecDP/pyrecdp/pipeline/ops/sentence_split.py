from .operator import OPERATORS, Operator
from ..utils.model_utils import prepare_model, MODEL_ZOO
from .common import get_sentences_from_document
from typing import Dict, Any
import numpy as np


@OPERATORS.register_module("sentence_split")
class SentenceSplit(Operator):
    def __init__(self, lang: str = 'en', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_key = prepare_model(lang, model_type="nltk")

    def processRow(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get sentences from a document.

        :param sample: document that need to split sentences
        :return: document with the sentences separated by '\\\\n'
        """
        nltk_model = MODEL_ZOO.get(self.model_key, None)
        tokenizer = nltk_model.tokenize if nltk_model else None
        sample[self.text_key] = get_sentences_from_document(sample[self.text_key], model_func=tokenizer)
        return sample

    def processBatch(self, batch: Dict[str,  np.ndarray]) -> Dict[str,  np.ndarray]:
        nltk_model = MODEL_ZOO.get(self.model_key, None)
        tokenizer = nltk_model.tokenize if nltk_model else None
        texts = []
        for text in batch[self.text_key]:
            texts.append(get_sentences_from_document(text, model_func=tokenizer))

        batch[self.text_key] = np.array(texts)
        return batch


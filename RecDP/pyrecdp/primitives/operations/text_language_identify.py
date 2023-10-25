from .base import BaseLLMOperation, LLMOPERATORS
from ray.data import Dataset
from pyspark.sql import DataFrame
from huggingface_hub import hf_hub_download
import fasttext
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
import inspect

class Transformer:
    parallelisable: bool = True
    expect_json: bool = False
    warn_when_pickling: bool = False
    ready: bool = False

    def __init_subclass__(cls, expect_json: bool = None):
        """Detects if the subclass expects json as input."""
        spec = inspect.getfullargspec(cls.do)
        if expect_json is None:
            expect_json = spec.annotations.get(spec.args[1], None) == dict

        cls.expect_json = expect_json

    def __new__(cls, *args, **kwargs):
        """Creates the transformer and save the arguments passed to the constructor."""
        t = super().__new__(cls)
        Transformer.__init__(t, args, kwargs)
        return t

    def __init__(self, state_args: tuple = None, state_kwargs: dict = None):
        """
        Init the transformer counters.

        If state_args/state_kwargs are set they will override whatever was
        originally passed to the subclass constructor.
        """
        if state_args is not None:
            self.__args = state_args
        if state_kwargs is not None:
            self.__kwargs = state_kwargs

        self.processed = 0
        self.__cls = type(self)

    def __call__(self, x):
        assert self.ready, f"{self} is not ready."
        if x is None:
            return
        y = self.do(x)
        self.processed += 1
        return y

    def do(self, x):
        raise NotImplementedError(f"'do' not implemented in {type(self)}")

    def map(self, source: Iterable) -> Iterator:
        if self.ready:
            for x in source:
                yield self(x)
            # since we have been prepared by caller,
            # caller is also responsible for calling `close`.
            return
        else:
            with self:
                for x in source:
                    yield self(x)

    def __getstate__(self) -> Tuple[tuple, dict, bool]:
        return (self.__args, self.__kwargs, self.expect_json)

    def __setstate__(self, state: Tuple[tuple, dict, bool]):
        if self.warn_when_pickling:
            print(f"Unpickling transformer: {type(self)}. This can be slow.")
        (args, kwargs, expect_json) = state
        # When unpickling `__new__` isn't called so we have to doit ourselves.
        Transformer.__init__(self, state_args=args, state_kwargs=kwargs)
        type(self).__init__(self, *args, **kwargs)
        assert self.expect_json == expect_json
        # __setstate__ is called by multiprocessing right before calling
        # the object so we need to initialize everything.
        self.__enter__()

    def _prepare(self) -> None:
        pass

    def __enter__(self) -> "Transformer":
        # In multiprocessing __enter__ is always called twice, so we are idempotent.
        # Because we call __enter__ when deserializing this transformer and
        # also when the parent transformer is deserialized.
        if self.ready:
            return self
        self._prepare()
        self.ready = True
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        pass


class Classifier(Transformer):
    def __init__(
        self,
        model: Path,
        field: str,
        out_field: str,
        threshold: float = 0,
        top: int = 1,
        language: str = None,
        rounding: int = 2,
    ):
        super().__init__()
        self.model = model
        assert model.exists(), f"Model {model} doesn't exist."
        self.field = field
        self.out_field = out_field
        self.threshold = threshold
        self.top = top
        self.language = language
        self.rounding = rounding
        # Fasttext model is a C object and can't be pickled
        self.fasttext_model: fasttext._FastText = None
        self.n_doc, self.n_accepted, self.n_ignored, self.n_disagreement = 0, 0, 0, 0
        self.cnt: Dict[str, int] = {}

    def _prepare(self):
        self.fasttext_model = fasttext.load_model(str(self.model))

    def predict(self, text):
        return predict(self.fasttext_model, text.replace("\n", ""), k=self.top)

    def do(self, doc: dict) -> Optional[dict]:
        text = doc.get(self.field, None)
        if not text:
            return None

        if self.language and doc.get("language") != self.language:
            self.n_ignored += 1
            return doc

        self.n_doc += 1
        labels, scores = self.predict(text)
        scores.round(self.rounding, out=scores)
        for l in labels:
            self.cnt[l] = self.cnt.get(l, 0) + 1

        if self.top == 1:
            existing_label = doc.get(self.out_field, None)
            if existing_label and labels[0] != existing_label:
                self.n_disagreement += 1

        if all(s < self.threshold for s in scores):
            return None

        self.n_accepted += 1
        if self.top == 1:
            doc[self.out_field] = labels[0]
            doc[self.out_field + "_score"] = scores[0]
        else:
            doc[self.out_field] = {l: s for l, s in zip(labels, scores)}
        return doc

    def __repr__(self):
        return f"Classifier({self.model})"

  
def predict(model, text: str, k: int = 1):
    labels, scores = model.predict(text, k=k)
    labels = [l.replace("__label__", "") for l in labels]
    return labels, scores

def construct_classifier(fasttext_model_dir, language_identify_field, language_identify_output_field, threshold):
    print(fasttext_model_dir)
    if os.path.isfile(fasttext_model_dir):
        model_path = fasttext_model_dir
    else:
        model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
    model = Path(model_path)
    return Classifier(model, language_identify_field, language_identify_output_field, threshold=threshold)

def prepare_func_language_id(fasttext_model_dir, language_identify_field, language_identify_output_field, threshold):
    classifier = construct_classifier(fasttext_model_dir, language_identify_field, language_identify_output_field, threshold)
    def generate_lang_label(content):
        # 0. apply normalization to content
        content = {classifier.field: content}
        content =classifier(content)
        return content[classifier.out_field] if content else ""
    return generate_lang_label

class LanguageIdentify(BaseLLMOperation):
    def __init__(self, text_key = 'text', inplace = False, fasttext_model_dir = "", threshold = 0):
        """
        Initialization method
        :param text_key: the name of field which will be apply language_idenfity.
        :param fasttext_model_dir: The min language identification confidence
            scores of samples to keep.
        :param fasttext_model_dir: the path of fasttext model dir. if fasttext_model_dir equals to "", the code will download the model from huggingface
        :param threshold: the threshold to return identified language. the value range is [0, 1).
        """
        settings = {'text_key': text_key, 'inplace': inplace, 'fasttext_model_dir': fasttext_model_dir, 'threshold': threshold}
        super().__init__(settings)
        self.text_key = text_key
        self.inplace = False
        self.fasttext_model_dir = fasttext_model_dir
        self.threshold = threshold
        self.actual_func = None
        self.support_spark = True
        self.support_ray = True
        
    def process_rayds(self, ds: Dataset) -> Dataset:
        if self.inplace:
            raise NotImplementedError("We only support non-inplace modification for LanguageIdentify.")
        else:
            new_name = 'language'
        if self.actual_func is None:
            self.actual_func = prepare_func_language_id(fasttext_model_dir = self.fasttext_model_dir,
                                                        language_identify_field = self.text_key, language_identify_output_field = new_name, threshold = self.threshold)
        return ds.map(lambda x: self.process_row(x, self.text_key, new_name, self.actual_func))
    
    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        import pyspark.sql.functions as F
        from pyspark.sql import types as T
        new_name = "language"
        language_id_udf = F.udf(prepare_func_language_id(fasttext_model_dir = self.fasttext_model_dir,
                                                         language_identify_field = self.text_key, language_identify_output_field = new_name, threshold = self.threshold))
        return spark_df.withColumn(new_name, language_id_udf(F.col(self.text_key)))
    
LLMOPERATORS.register(LanguageIdentify)

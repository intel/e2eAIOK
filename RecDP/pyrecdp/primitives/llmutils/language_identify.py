import argparse
import fasttext
import inspect
import jsonlines
import json
import logging
import time
import warnings
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
)

from pyrecdp.core.utils import Timer
from pyrecdp.primitives.llmutils.utils import *

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

        self.start_time = time.time()
        self.__last_log = self.start_time
        self.processed = 0
        # Log every 5 min unless specified other wise.
        self._log_freq = int(os.environ.get("JSONQL_LOG_FREQ", 5 * 60))
        self.__cls = type(self)
        self._logger = logging.getLogger(self.__cls.__name__)

    def __call__(self, x):
        assert self.ready, f"{self} is not ready."
        if x is None:
            return
        y = self.do(x)
        self.processed += 1
        if time.time() - self.__last_log > self._log_freq:
            self.log_summary()
        return y

    def do(self, x):
        raise NotImplementedError(f"'do' not implemented in {type(self)}")

    def summary(self) -> List[str]:
        return [self.speed_summary()]

    def speed_summary(self) -> str:
        delay = time.time() - self.start_time
        h = delay / 3600
        s = self.processed / delay
        return f"Processed {self.processed:_} documents in {h:.2}h ({s:5.1f} doc/s)."

    def log(self, message):
        self._logger.info(message)

    def log_summary(self) -> None:
        if not self.ready:
            self.log("Not ready.")
            return
        summ = self.summary() or []
        for line in summ:
            self.log(line)
        self.__last_log = time.time()

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
            warnings.warn(f"Unpickling transformer: {type(self)}. This can be slow.")
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
        self.start_time = time.time()
        if self.ready:
            return self
        self._prepare()
        self.ready = True
        return self

    def __exit__(self, *args) -> None:
        self.close()
        self.log_summary()

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
        self.log(f"Loading {self.model}")
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

    def summary(self):
        n_doc, n_accepted, n_disagreement, cnt, out_field = (
            self.n_doc,
            self.n_accepted,
            self.n_disagreement,
            self.cnt,
            self.out_field,
        )
        summ = super().summary()
        if self.threshold > 0:
            ratio = n_accepted / n_doc if n_doc else 0
            summ.append(f"Kept {n_accepted} docs over {n_doc} ({ratio :.1%})")
        summ.append(f"Found {len(cnt)} {out_field} labels: {cnt}")

        disagreement = n_disagreement / n_doc if n_doc else 0
        if disagreement:
            summ.append(f"{out_field} disagreement is at {disagreement:.1%}.")
        return summ

    def __repr__(self):
        return f"Classifier({self.model})"


def predict(model, text: str, k: int = 1):
    labels, scores = model.predict(text, k=k)
    labels = [l.replace("__label__", "") for l in labels]
    return labels, scores


def multi_run_language_identify(args):
   return language_identify(*args)


def language_identify(x_list, filter_condition):
    for x in x_list:
        in_file_name, out_file_name = x
        with open(in_file_name, 'r') as rdr:
            with jsonlines.open(out_file_name, 'w') as f:
                for line in rdr:
                    try:
                        json_line = json.loads(line)
                        if filter_condition(json_line):
                            f.write(json_line)
                    except json.decoder.JSONDecodeError:
                        print("Faild to load json data, Skip this line.")


def language_identify_MP(data_dir, language_identify_filter, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(os.listdir(data_dir))
    files = list(filter(lambda file_: '.jsonl' in file_, files))

    args = [(os.path.join(data_dir, i), os.path.join(out_dir, i)) for i in files]

    n_chunks, n_proc = get_nchunks_and_nproc(len(files))
    print(f"resetting to {n_proc} for number of processes")

    args = [(args[i: i + n_chunks], language_identify_filter) for i in range(0, len(args), n_chunks)]

    launch_mp(n_proc, args, multi_run_language_identify)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest="data_dir", type=str)
    parser.add_argument("--fasttext_model", dest="fasttext_model", type=str)
    parser.add_argument("--language", dest="language", type=str, default="")
    parser.add_argument("--language_identify_output", dest="language_identify_output", type=str, default="")
    parser.add_argument("--language_identify_field", dest="language_identify_field", type=str, default="text")
    parser.add_argument("--language_identify_output_field", dest="language_identify_output_field", type=str, default="lang")

    args = parser.parse_args()
    data_dir = args.data_dir
    data_files = get_data_files(data_dir)

    fasttext_model = args.fasttext_model
    target_language = args.language
    language_identify_output = os.path.join(data_dir, "language_identify") \
        if args.language_identify_output == "" else args.language_identify_output
    language_identify_field = args.language_identify_field
    language_identify_output_field = args.language_identify_output_field

    model = Path(fasttext_model)
    if not model.exists():
        exit(1)
    classifier = Classifier(model, language_identify_field, language_identify_output_field)

    def language_identify_filter(content):
        classifier.__enter__()
        identify_language = classifier(content)[classifier.out_field]

        return True if identify_language == target_language or target_language == "" else False

    with Timer(f"Generate language_identify data for {data_dir}"):
        language_identify_MP(data_dir, language_identify_filter, language_identify_output)

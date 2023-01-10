import torch
from joblib import Parallel, delayed
from e2eAIOK.DeNas.asr.utils.data_utils import undo_padding
from e2eAIOK.DeNas.asr.utils.edit_distance import wer_summary, wer_details_for_batch
from e2eAIOK.DeNas.asr.data.dataio.dataio import merge_char, split_word
from e2eAIOK.DeNas.asr.data.dataio.wer import print_wer_summary, print_alignments


class MetricStats:
    """A default class for storing and summarizing arbitrary metrics.

    More complex metrics can be created by sub-classing this class.

    Arguments
    ---------
    metric : function
        The function to use to compute the relevant metric. Should take
        at least two arguments (predictions and targets) and can
        optionally take the relative lengths of either or both arguments.
        Not usually used in sub-classes.
    batch_eval: bool
        When True it feeds the evaluation metric with the batched input.
        When False and n_jobs=1, it performs metric evaluation one-by-one
        in a sequential way. When False and n_jobs>1, the evaluation
        runs in parallel over the different inputs using joblib.
    n_jobs : int
        The number of jobs to use for computing the metric. If this is
        more than one, every sample is processed individually, otherwise
        the whole batch is passed at once.
    """

    def __init__(self, metric, n_jobs=1, batch_eval=True):
        self.metric = metric
        self.n_jobs = n_jobs
        self.batch_eval = batch_eval
        self.clear()

    def clear(self):
        """Creates empty container for storage, removing existing stats."""
        self.scores = []
        self.ids = []
        self.summary = {}

    def append(self, ids, *args, **kwargs):
        """Store a particular set of metric scores.

        Arguments
        ---------
        ids : list
            List of ids corresponding to utterances.
        *args, **kwargs
            Arguments to pass to the metric function.
        """
        self.ids.extend(ids)

        # Batch evaluation
        if self.batch_eval:
            scores = self.metric(*args, **kwargs).detach()

        else:
            if "predict" not in kwargs or "target" not in kwargs:
                raise ValueError(
                    "Must pass 'predict' and 'target' as kwargs if batch_eval=False"
                )
            if self.n_jobs == 1:
                # Sequence evaluation (loop over inputs)
                scores = sequence_evaluation(metric=self.metric, **kwargs)
            else:
                # Multiprocess evaluation
                scores = multiprocess_evaluation(
                    metric=self.metric, n_jobs=self.n_jobs, **kwargs
                )

        self.scores.extend(scores)

    def summarize(self, field=None):
        """Summarize the metric scores, returning relevant stats.

        Arguments
        ---------
        field : str
            If provided, only returns selected statistic. If not,
            returns all computed statistics.

        Returns
        -------
        float or dict
            Returns a float if ``field`` is provided, otherwise
            returns a dictionary containing all computed stats.
        """
        min_index = torch.argmin(torch.tensor(self.scores))
        max_index = torch.argmax(torch.tensor(self.scores))
        self.summary = {
            "average": float(sum(self.scores) / len(self.scores)),
            "min_score": float(self.scores[min_index]),
            "min_id": self.ids[min_index],
            "max_score": float(self.scores[max_index]),
            "max_id": self.ids[max_index],
        }

        if field is not None:
            return self.summary[field]
        else:
            return self.summary

    def write_stats(self, filestream, verbose=False):
        """Write all relevant statistics to file.

        Arguments
        ---------
        filestream : file-like object
            A stream for the stats to be written to.
        verbose : bool
            Whether to also print the stats to stdout.
        """
        if not self.summary:
            self.summarize()

        message = f"Average score: {self.summary['average']}\n"
        message += f"Min error: {self.summary['min_score']} "
        message += f"id: {self.summary['min_id']}\n"
        message += f"Max error: {self.summary['max_score']} "
        message += f"id: {self.summary['max_id']}\n"

        filestream.write(message)
        if verbose:
            print(message)


def multiprocess_evaluation(metric, predict, target, lengths=None, n_jobs=8):
    """Runs metric evaluation if parallel over multiple jobs."""
    if lengths is not None:
        lengths = (lengths * predict.size(1)).round().int().cpu()
        predict = [p[:length].cpu() for p, length in zip(predict, lengths)]
        target = [t[:length].cpu() for t, length in zip(target, lengths)]

    while True:
        try:
            scores = Parallel(n_jobs=n_jobs, timeout=30)(
                delayed(metric)(p, t) for p, t in zip(predict, target)
            )
            break
        except Exception as e:
            print(e)
            print("Evaluation timeout...... (will try again)")

    return scores


def sequence_evaluation(metric, predict, target, lengths=None):
    """Runs metric evaluation sequentially over the inputs."""
    if lengths is not None:
        lengths = (lengths * predict.size(1)).round().int().cpu()
        predict = [p[:length].cpu() for p, length in zip(predict, lengths)]
        target = [t[:length].cpu() for t, length in zip(target, lengths)]

    scores = []
    for p, t in zip(predict, target):
        score = metric(p, t)
        scores.append(score)
    return scores


class ErrorRateStats(MetricStats):
    """A class for tracking error rates (e.g., WER, PER).

    Arguments
    ---------
    merge_tokens : bool
        Whether to merge the successive tokens (used for e.g.,
        creating words out of character tokens).
    split_tokens : bool
        Whether to split tokens (used for e.g. creating
        characters out of word tokens).
    space_token : str
        The character to use for boundaries. Used with ``merge_tokens``
        this represents character to split on after merge.
        Used with ``split_tokens`` the sequence is joined with
        this token in between, and then the whole sequence is split.
    """

    def __init__(self, merge_tokens=False, split_tokens=False, space_token="_"):
        self.clear()
        self.merge_tokens = merge_tokens
        self.split_tokens = split_tokens
        self.space_token = space_token

    def append(
        self,
        ids,
        predict,
        target,
        predict_len=None,
        target_len=None,
        ind2lab=None,
    ):
        """Add stats to the relevant containers.

        * See MetricStats.append()

        Arguments
        ---------
        ids : list
            List of ids corresponding to utterances.
        predict : torch.tensor
            A predicted output, for comparison with the target output
        target : torch.tensor
            The correct reference output, for comparison with the prediction.
        predict_len : torch.tensor
            The predictions relative lengths, used to undo padding if
            there is padding present in the predictions.
        target_len : torch.tensor
            The target outputs' relative lengths, used to undo padding if
            there is padding present in the target.
        ind2lab : callable
            Callable that maps from indices to labels, operating on batches,
            for writing alignments.
        """
        self.ids.extend(ids)

        if predict_len is not None:
            predict = undo_padding(predict, predict_len)

        if target_len is not None:
            target = undo_padding(target, target_len)

        if ind2lab is not None:
            predict = ind2lab(predict)
            target = ind2lab(target)

        if self.merge_tokens:
            predict = merge_char(predict, space=self.space_token)
            target = merge_char(target, space=self.space_token)

        if self.split_tokens:
            predict = split_word(predict, space=self.space_token)
            target = split_word(target, space=self.space_token)

        scores = wer_details_for_batch(ids, target, predict, True)

        self.scores.extend(scores)

    def summarize(self, field=None):
        """Summarize the error_rate and return relevant statistics.

        * See MetricStats.summarize()
        """
        self.summary = wer_summary(self.scores)

        # Add additional, more generic key
        self.summary["error_rate"] = self.summary["WER"]

        if field is not None:
            return self.summary[field]
        else:
            return self.summary

    def write_stats(self, filestream):
        """Write all relevant info (e.g., error rate alignments) to file.
        * See MetricStats.write_stats()
        """
        if not self.summary:
            self.summarize()

        print_wer_summary(self.summary, filestream)
        print_alignments(self.scores, filestream)

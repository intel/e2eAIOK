import torch
import logging
import functools
from asr.data.dataio.dataio import length_to_mask


logger = logging.getLogger(__name__)


def ctc_loss(
    log_probs, targets, input_lens, target_lens, blank_index, reduction="mean"
):
    """CTC loss.

    Arguments
    ---------
    predictions : torch.Tensor
        Predicted tensor, of shape [batch, time, chars].
    targets : torch.Tensor
        Target tensor, without any blanks, of shape [batch, target_len]
    input_lens : torch.Tensor
        Length of each utterance.
    target_lens : torch.Tensor
        Length of each target sequence.
    blank_index : int
        The location of the blank symbol among the character indexes.
    reduction : str
        What reduction to apply to the output. 'mean', 'sum', 'batch',
        'batchmean', 'none'.
        See pytorch for 'mean', 'sum', 'none'. The 'batch' option returns
        one loss per item in the batch, 'batchmean' returns sum / batch size.
    """
    input_lens = (input_lens * log_probs.shape[1]).round().int()
    target_lens = (target_lens * targets.shape[1]).round().int()
    log_probs = log_probs.transpose(0, 1)

    if reduction == "batchmean":
        reduction_loss = "sum"
    elif reduction == "batch":
        reduction_loss = "none"
    else:
        reduction_loss = reduction
    loss = torch.nn.functional.ctc_loss(
        log_probs,
        targets,
        input_lens,
        target_lens,
        blank_index,
        zero_infinity=True,
        reduction=reduction_loss,
    )

    if reduction == "batchmean":
        return loss / targets.shape[0]
    elif reduction == "batch":
        N = loss.size(0)
        return loss.view(N, -1).sum(1) / target_lens.view(N, -1).sum(1)
    else:
        return loss


def nll_loss(
    log_probabilities,
    targets,
    length=None,
    label_smoothing=0.0,
    allowed_len_diff=3,
    reduction="mean",
):
    """Computes negative log likelihood loss.

    Arguments
    ---------
    log_probabilities : torch.Tensor
        The probabilities after log has been applied.
        Format is [batch, log_p] or [batch, frames, log_p].
    targets : torch.Tensor
        The targets, of shape [batch] or [batch, frames].
    length : torch.Tensor
        Length of each utterance, if frame-level loss is desired.
    allowed_len_diff : int
        Length difference that will be tolerated before raising an exception.
    reduction : str
        Options are 'mean', 'batch', 'batchmean', 'sum'.
        See pytorch for 'mean', 'sum'. The 'batch' option returns
        one loss per item in the batch, 'batchmean' returns sum / batch size.

    Example
    -------
    >>> probs = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    >>> nll_loss(torch.log(probs), torch.tensor([1, 1]))
    tensor(1.2040)
    """
    if len(log_probabilities.shape) == 3:
        log_probabilities, targets = truncate(
            log_probabilities, targets, allowed_len_diff
        )
        log_probabilities = log_probabilities.transpose(1, -1)

    # Pass the loss function but apply reduction="none" first
    loss = functools.partial(torch.nn.functional.nll_loss, reduction="none")
    return compute_masked_loss(
        loss,
        log_probabilities,
        targets.long(),
        length,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )


def kldiv_loss(
    log_probabilities,
    targets,
    length=None,
    label_smoothing=0.0,
    allowed_len_diff=3,
    pad_idx=0,
    reduction="mean",
):
    """Computes the KL-divergence error at the batch level.
    This loss applies label smoothing directly to the targets

    Arguments
    ---------
    probabilities : torch.Tensor
        The posterior probabilities of shape
        [batch, prob] or [batch, frames, prob].
    targets : torch.Tensor
        The targets, of shape [batch] or [batch, frames].
    length : torch.Tensor
        Length of each utterance, if frame-level loss is desired.
    allowed_len_diff : int
        Length difference that will be tolerated before raising an exception.
    reduction : str
        Options are 'mean', 'batch', 'batchmean', 'sum'.
        See pytorch for 'mean', 'sum'. The 'batch' option returns
        one loss per item in the batch, 'batchmean' returns sum / batch size.

    Example
    -------
    >>> probs = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    >>> kldiv_loss(torch.log(probs), torch.tensor([1, 1]))
    tensor(1.2040)
    """
    if label_smoothing > 0:
        if log_probabilities.dim() == 2:
            log_probabilities = log_probabilities.unsqueeze(1)

        bz, time, n_class = log_probabilities.shape
        targets = targets.long().detach()

        confidence = 1 - label_smoothing

        log_probabilities = log_probabilities.view(-1, n_class)
        targets = targets.view(-1)
        with torch.no_grad():
            true_distribution = log_probabilities.clone()
            true_distribution.fill_(label_smoothing / (n_class - 1))
            ignore = targets == pad_idx
            targets = targets.masked_fill(ignore, 0)
            true_distribution.scatter_(1, targets.unsqueeze(1), confidence)

        loss = torch.nn.functional.kl_div(
            log_probabilities, true_distribution, reduction="none"
        )
        loss = loss.masked_fill(ignore.unsqueeze(1), 0)

        # return loss according to reduction specified
        if reduction == "mean":
            return loss.sum().mean()
        elif reduction == "batchmean":
            return loss.sum() / bz
        elif reduction == "batch":
            return loss.view(bz, -1).sum(1) / length
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss
    else:
        return nll_loss(log_probabilities, targets, length, reduction=reduction)


def truncate(predictions, targets, allowed_len_diff=3):
    """Ensure that predictions and targets are the same length.

    Arguments
    ---------
    predictions : torch.Tensor
        First tensor for checking length.
    targets : torch.Tensor
        Second tensor for checking length.
    allowed_len_diff : int
        Length difference that will be tolerated before raising an exception.
    """
    len_diff = predictions.shape[1] - targets.shape[1]
    if len_diff == 0:
        return predictions, targets
    elif abs(len_diff) > allowed_len_diff:
        raise ValueError(
            "Predictions and targets should be same length, but got %s and "
            "%s respectively." % (predictions.shape[1], targets.shape[1])
        )
    elif len_diff < 0:
        return predictions, targets[:, : predictions.shape[1]]
    else:
        return predictions[:, : targets.shape[1]], targets


def compute_masked_loss(
    loss_fn,
    predictions,
    targets,
    length=None,
    label_smoothing=0.0,
    reduction="mean",
):
    """Compute the true average loss of a set of waveforms of unequal length.

    Arguments
    ---------
    loss_fn : function
        A function for computing the loss taking just predictions and targets.
        Should return all the losses, not a reduction (e.g. reduction="none").
    predictions : torch.Tensor
        First argument to loss function.
    targets : torch.Tensor
        Second argument to loss function.
    length : torch.Tensor
        Length of each utterance to compute mask. If None, global average is
        computed and returned.
    label_smoothing: float
        The proportion of label smoothing. Should only be used for NLL loss.
        Ref: Regularizing Neural Networks by Penalizing Confident Output
        Distributions. https://arxiv.org/abs/1701.06548
    reduction : str
        One of 'mean', 'batch', 'batchmean', 'none' where 'mean' returns a
        single value and 'batch' returns one per item in the batch and
        'batchmean' is sum / batch_size and 'none' returns all.
    """
    mask = torch.ones_like(targets)
    if length is not None:
        length_mask = length_to_mask(
            length * targets.shape[1], max_len=targets.shape[1],
        )

        # Handle any dimensionality of input
        while len(length_mask.shape) < len(mask.shape):
            length_mask = length_mask.unsqueeze(-1)
        length_mask = length_mask.type(mask.dtype)
        mask *= length_mask

    # Compute, then reduce loss
    loss = loss_fn(predictions, targets) * mask
    N = loss.size(0)
    if reduction == "mean":
        loss = loss.sum() / torch.sum(mask)
    elif reduction == "batchmean":
        loss = loss.sum() / N
    elif reduction == "batch":
        loss = loss.reshape(N, -1).sum(1) / mask.reshape(N, -1).sum(1)

    if label_smoothing == 0:
        return loss
    else:
        loss_reg = torch.mean(predictions, dim=1) * mask
        if reduction == "mean":
            loss_reg = torch.sum(loss_reg) / torch.sum(mask)
        elif reduction == "batchmean":
            loss_reg = torch.sum(loss_reg) / targets.shape[0]
        elif reduction == "batch":
            loss_reg = loss_reg.sum(1) / mask.sum(1)

        return -label_smoothing * loss_reg + (1 - label_smoothing) * loss

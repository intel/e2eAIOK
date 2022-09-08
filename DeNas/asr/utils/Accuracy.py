import torch
from asr.data.dataio.dataio import length_to_mask


def Accuracy(log_probabilities, targets, length=None):
    """Calculates the accuracy for predicted log probabilities and targets in a batch.

    Arguments
    ----------
    log_probabilities : tensor
        Predicted log probabilities (batch_size, time, feature).
    targets : tensor
        Target (batch_size, time).
    length : tensor
        Length of target (batch_size,).
    """
    if length is not None:
        mask = length_to_mask(
            length * targets.shape[1], max_len=targets.shape[1],
        ).bool()
        if len(targets.shape) == 3:
            mask = mask.unsqueeze(2).repeat(1, 1, targets.shape[2])

    padded_pred = log_probabilities.argmax(-1)

    if length is not None:
        numerator = torch.sum(
            padded_pred.masked_select(mask) == targets.masked_select(mask)
        )
        denominator = torch.sum(mask)
    else:
        numerator = torch.sum(padded_pred == targets)
        denominator = targets.shape[1]
    return float(numerator), float(denominator)


class AccuracyStats:
    """Module for calculate the overall one-step-forward prediction accuracy.
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def append(self, log_probabilities, targets, length=None):
        """This function is for updating the stats according to the prediction
        and target in the current batch.

        Arguments
        ----------
        log_probabilities : tensor
            Predicted log probabilities (batch_size, time, feature).
        targets : tensor
            Target (batch_size, time).
        length: tensor
            Length of target (batch_size,).
        """
        numerator, denominator = Accuracy(log_probabilities, targets, length)
        self.correct += numerator
        self.total += denominator

    def summarize(self):
        """Computes the accuract metric."""
        return self.correct / self.total

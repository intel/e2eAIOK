import torch
import logging

from asr.utils import checkpoints

logger = logging.getLogger(__name__)


@checkpoints.register_checkpoint_hooks
class NoamScheduler:
    """The is an implementation of the transformer's learning rate scheduler with warmup.
    Reference: https://arxiv.org/abs/1706.03762

    Note: this scheduler anneals the lr at each update of the model's weight,
    and n_steps must be saved for restarting.

    Arguments
    ---------
    lr_initial : float
        Initial learning rate (i.e. the lr used at epoch 0).
    n_warmup_steps : int
        numer of warm-up steps
    model_size : int
        size of transformer embed_dim. It is used to scale the maximum learning rate value reached
        by the scheduler. It is divided by model_size ** (0.5).
        If not specified the maximum learning rate value is instead multiplied by warmup_steps ** (0.5).
    """

    def __init__(self, lr_initial, n_warmup_steps, model_size=None):
        self.lr_initial = lr_initial
        self.n_warmup_steps = n_warmup_steps
        self.current_lr = lr_initial
        self.losses = []
        self.n_steps = 0
        self.normalize = n_warmup_steps ** 0.5
        if model_size is not None:
            self.normalize = model_size ** (-0.5)

    def __call__(self, opt):
        """
        Arguments
        ---------
        opt : optimizer
            The optimizer to update using this scheduler.

        Returns
        -------
        current_lr : float
            The learning rate before the update.
        lr : float
            The learning rate after the update.
        """
        self.n_steps += 1

        current_lr = opt.param_groups[0]["lr"]

        lr = self.lr_initial * self._get_lr_scale()

        # Changing the learning rate within the optimizer
        for param_group in opt.param_groups:
            param_group["lr"] = lr

        self.current_lr = current_lr
        return current_lr, lr

    def _get_lr_scale(self):
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return self.normalize * min(
            n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5)
        )

    @checkpoints.mark_as_saver
    def save(self, path):
        """Saves the current metrics on the specified path."""
        data = {"losses": self.losses, "n_steps": self.n_steps}
        torch.save(data, path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch=False, device=None):
        """Loads the needed information."""
        del end_of_epoch  # Unused in this class
        del device
        data = torch.load(path)
        self.losses = data["losses"]
        self.n_steps = data["n_steps"]

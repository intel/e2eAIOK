import collections
import torch
from torch.utils.data._utils.collate import default_convert
from torch.utils.data._utils.pin_memory import (
    pin_memory as recursive_pin_memory,
)

from e2eAIOK.DeNas.asr.utils.data_utils import mod_default_collate
from e2eAIOK.DeNas.asr.utils.data_utils import recursive_to
from e2eAIOK.DeNas.asr.utils.data_utils import batch_pad_right


PaddedData = collections.namedtuple("PaddedData", ["data", "lengths"])


class PaddedBatch:
    """Collate_fn when examples are dicts and have variable-length sequences.

    Different elements in the examples get matched by key.
    All numpy tensors get converted to Torch (PyTorch default_convert)
    Then, by default, all torch.Tensor valued elements get padded and support
    collective pin_memory() and to() calls.
    Regular Python data types are just collected in a list.

    Arguments
    ---------
    examples : list
        List of example dicts, as produced by Dataloader.
    padded_keys : list, None
        (Optional) List of keys to pad on. If None, pad all torch.Tensors
    device_prep_keys : list, None
        (Optional) Only these keys participate in collective memory pinning and moving with
        to().
        If None, defaults to all items with torch.Tensor values.
    padding_func : callable, optional
        Called with a list of tensors to be padded together. Needs to return
        two tensors: the padded data, and another tensor for the data lengths.
    padding_kwargs : dict
        (Optional) Extra kwargs to pass to padding_func. E.G. mode, value
    apply_default_convert : bool
        Whether to apply PyTorch default_convert (numpy to torch recursively,
        etc.) on all data. Default:True, usually does the right thing.
    nonpadded_stack : bool
        Whether to apply PyTorch-default_collate-like stacking on values that
        didn't get padded. This stacks if it can, but doesn't error out if it
        cannot. Default:True, usually does the right thing.
    """

    def __init__(
        self,
        examples,
        padded_keys=None,
        device_prep_keys=None,
        padding_func=batch_pad_right,
        padding_kwargs={},
        apply_default_convert=True,
        nonpadded_stack=True,
    ):
        self.__length = len(examples)
        self.__keys = list(examples[0].keys())
        self.__padded_keys = []
        self.__device_prep_keys = []
        for key in self.__keys:
            values = [example[key] for example in examples]
            # Default convert usually does the right thing (numpy2torch etc.)
            if apply_default_convert:
                values = default_convert(values)
            if (padded_keys is not None and key in padded_keys) or (
                padded_keys is None and isinstance(values[0], torch.Tensor)
            ):
                # Padding and PaddedData
                self.__padded_keys.append(key)
                padded = PaddedData(*padding_func(values, **padding_kwargs))
                setattr(self, key, padded)
            else:
                # Default PyTorch collate usually does the right thing
                # (convert lists of equal sized tensors to batch tensors, etc.)
                if nonpadded_stack:
                    values = mod_default_collate(values)
                setattr(self, key, values)
            if (device_prep_keys is not None and key in device_prep_keys) or (
                device_prep_keys is None and isinstance(values[0], torch.Tensor)
            ):
                self.__device_prep_keys.append(key)

    def __len__(self):
        return self.__length

    def __getitem__(self, key):
        if key in self.__keys:
            return getattr(self, key)
        else:
            raise KeyError(f"Batch doesn't have key: {key}")

    def __iter__(self):
        """Iterates over the different elements of the batch.
        """
        return iter((getattr(self, key) for key in self.__keys))

    def pin_memory(self):
        """In-place, moves relevant elements to pinned memory."""
        for key in self.__device_prep_keys:
            value = getattr(self, key)
            pinned = recursive_pin_memory(value)
            setattr(self, key, pinned)
        return self

    def to(self, *args, **kwargs):
        """In-place move/cast relevant elements.

        Passes all arguments to torch.Tensor.to.
        """
        for key in self.__device_prep_keys:
            value = getattr(self, key)
            moved = recursive_to(value, *args, **kwargs)
            setattr(self, key, moved)
        return self

    def at_position(self, pos):
        """Gets the position."""
        key = self.__keys[pos]
        return getattr(self, key)

    @property
    def batchsize(self):
        """Returns the bach size"""
        return self.__length

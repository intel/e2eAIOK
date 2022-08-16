import torch
import logging


logger = logging.getLogger(__name__)

def load_torch_model(obj, path, device):
    incompatible_keys = obj.load_state_dict(
        torch.load(path, map_location=device), strict=False
    )
    for missing_key in incompatible_keys.missing_keys:
        logger.warning(
            f"During parameter transfer to {obj} loading from "
            + f"{path}, the transferred parameters did not have "
            + f"parameters for the key: {missing_key}"
        )
    for unexpected_key in incompatible_keys.unexpected_keys:
        logger.warning(
            f"During parameter transfer to {obj} loading from "
            + f"{path}, the object could not use the parameters loaded "
            + f"with the key: {unexpected_key}"
        )

def load_spm(obj, path):
    obj.load(str(path))
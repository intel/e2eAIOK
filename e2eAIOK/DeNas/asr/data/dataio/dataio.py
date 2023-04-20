import os
import torch
import logging
import numpy as np
import pickle #nosec
import csv
import time
import torchaudio
import json
import re
logger = logging.getLogger(__name__)


def load_data_json(json_path, replacements={}):
    """Loads JSON and recursively formats string values.

    Arguments
    ----------
    json_path : str
        Path to CSV file.
    replacements : dict
        (Optional dict), This is used to recursively format all string values in the data.

    Returns
    -------
    dict
        JSON data with replacements applied.
    """
    with open(json_path, "r") as f:
        out_json = json.load(f)
    _recursive_format(out_json, replacements)
    return out_json


def _recursive_format(data, replacements):
    if isinstance(data, dict):
        for key, item in data.items():
            if isinstance(item, dict) or isinstance(item, list):
                _recursive_format(item, replacements)
            elif isinstance(item, str):
                data[key] = item.format_map(replacements)
            # If not dict, list or str, do nothing
    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict) or isinstance(item, list):
                _recursive_format(item, replacements)
            elif isinstance(item, str):
                data[i] = item.format_map(replacements)
            # If not dict, list or str, do nothing


def load_data_csv(csv_path, replacements={}):
    """Loads CSV and formats string values.

    CSV must have an 'ID' field.
    If there is a field called duration, it is interpreted as a float.
    The rest of the fields are left as they are (legacy _format and _opts fields
    are not used to load the data in any special way).

    Bash-like string replacements with $to_replace are supported.

    Arguments
    ----------
    csv_path : str
        Path to CSV file.
    replacements : dict
        (Optional dict), This is used to recursively format all string values in the data.

    Returns
    -------
    dict
        CSV data with replacements applied.
    """

    with open(csv_path, newline="") as csvfile:
        result = {}
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        variable_finder = re.compile(r"\$([\w.]+)")
        for row in reader:
            # ID:
            try:
                data_id = row["ID"]
                del row["ID"]  # This is used as a key in result, instead.
            except KeyError:
                raise KeyError(
                    "CSV has to have an 'ID' field, with unique ids"
                    " for all data points"
                )
            if data_id in result:
                raise ValueError(f"Duplicate id: {data_id}")
            # Replacements:
            for key, value in row.items():
                try:
                    row[key] = variable_finder.sub(
                        lambda match: str(replacements[match[1]]), value
                    )
                except KeyError:
                    raise KeyError(
                        f"The item {value} requires replacements "
                        "which were not supplied."
                    )
            # Duration:
            if "duration" in row:
                row["duration"] = float(row["duration"])
            result[data_id] = row
    return result


def read_audio(waveforms_obj):
    """General audio loading, based on a custom notation.

    Arguments
    ----------
    waveforms_obj : str, dict
        Audio reading annotation, see above for format.

    Returns
    -------
    torch.Tensor
        Audio tensor with shape: (samples, ).
    """
    if isinstance(waveforms_obj, str):
        audio, _ = torchaudio.load(waveforms_obj)
        return audio.transpose(0, 1).squeeze(1)

    path = waveforms_obj["file"]
    start = waveforms_obj.get("start", 0)
    # Default stop to start -> if not specified, num_frames becomes 0,
    # which is the torchaudio default
    stop = waveforms_obj.get("stop", start)
    num_frames = stop - start
    audio, fs = torchaudio.load(path, num_frames=num_frames, frame_offset=start)
    audio = audio.transpose(0, 1)
    return audio.squeeze(1)


def length_to_mask(length, max_len=None, dtype=None, device=None):
    """Creates a binary mask for each sequence.

    Reference: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397/3

    Arguments
    ---------
    length : torch.LongTensor
        Containing the length of each sequence in the batch. Must be 1D.
    max_len : int
        Max length for the mask, also the size of the second dimension.
    dtype : torch.dtype, default: None
        The dtype of the generated mask.
    device: torch.device, default: None
        The device to put the mask variable.

    Returns
    -------
    mask : tensor
        The binary mask.
    """
    assert len(length.shape) == 1

    if max_len is None:
        max_len = length.max().long().item()  # using arange to generate mask
    mask = torch.arange(
        max_len, device=length.device, dtype=length.dtype
    ).expand(len(length), max_len) < length.unsqueeze(1)

    if dtype is None:
        dtype = length.dtype

    if device is None:
        device = length.device

    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask


def save_pkl(obj, file):
    """Save an object in pkl format.

    Arguments
    ---------
    obj : object
        Object to save in pkl format
    file : str
        Path to the output file
    sampling_rate : int
        Sampling rate of the audio file, TODO: this is not used?
    """
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(file):
    """Loads a pkl file.

    Arguments
    ---------
    file : str
        Path to the input pkl file.

    Returns
    -------
    The loaded object.
    """

    # Deals with the situation where two processes are trying
    # to access the same label dictionary by creating a lock
    count = 100
    while count > 0:
        if os.path.isfile(file + ".lock"):
            time.sleep(1)
            count -= 1
        else:
            break

    try:
        with open(file + ".lock", "w"):
            pass
        with open(file, "rb") as f:
            return pickle.load(f) #nosec
    finally:
        if os.path.isfile(file + ".lock"):
            os.remove(file + ".lock")


def merge_char(sequences, space="_"):
    """Merge characters sequences into word sequences.

    Arguments
    ---------
    sequences : list
        Each item contains a list, and this list contains a character sequence.
    space : string
        The token represents space. Default: _

    Returns
    -------
    The list contains word sequences for each sentence.

    Example
    -------
    >>> sequences = [["a", "b", "_", "c", "_", "d", "e"], ["e", "f", "g", "_", "h", "i"]]
    >>> results = merge_char(sequences)
    >>> results
    [['ab', 'c', 'de'], ['efg', 'hi']]
    """
    results = []
    for seq in sequences:
        words = "".join(seq).split(space)
        results.append(words)
    return results


def merge_csvs(data_folder, csv_lst, merged_csv):
    """Merging several csv files into one file.

    Arguments
    ---------
    data_folder : string
        The folder to store csv files to be merged and after merging.
    csv_lst : list
        Filenames of csv file to be merged.
    merged_csv : string
        The filename to write the merged csv file.

    Example
    -------
    >>> merge_csvs("tests/samples/annotation/",
    ... ["speech.csv", "speech.csv"],
    ... "test_csv_merge.csv")
    """
    write_path = os.path.join(data_folder, merged_csv)
    if os.path.isfile(write_path):
        logger.info("Skipping merging. Completed in previous run.")
    with open(os.path.join(data_folder, csv_lst[0])) as f:
        header = f.readline()
    lines = []
    for csv_file in csv_lst:
        with open(os.path.join(data_folder, csv_file)) as f:
            for i, line in enumerate(f):
                if i == 0:
                    # Checking header
                    if line != header:
                        raise ValueError(
                            "Different header for " f"{csv_lst[0]} and {csv}."
                        )
                    continue
                lines.append(line)
    with open(write_path, "w") as f:
        f.write(header)
        for line in lines:
            f.write(line)
    logger.info(f"{write_path} is created.")


def split_word(sequences, space="_"):
    """Split word sequences into character sequences.

    Arguments
    ---------
    sequences : list
        Each item contains a list, and this list contains a words sequence.
    space : string
        The token represents space. Default: _

    Returns
    -------
    The list contains word sequences for each sentence.

    Example
    -------
    >>> sequences = [['ab', 'c', 'de'], ['efg', 'hi']]
    >>> results = split_word(sequences)
    >>> results
    [['a', 'b', '_', 'c', '_', 'd', 'e'], ['e', 'f', 'g', '_', 'h', 'i']]
    """
    results = []
    for seq in sequences:
        chars = list(space.join(seq))
        results.append(chars)
    return results

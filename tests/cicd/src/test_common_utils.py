import pytest
from e2eAIOK.common.utils import *

@pytest.mark.parametrize("path_input,expected", [("/home/vmagent/app/dataset/outbrain/train", "/home/vmagent/app/dataset/outbrain/train"), ("/home/vmagent/app/dataset/criteo/train", "/home/vmagent/app/dataset/criteo/train/train_data.bin")])
def test_get_file(path_input, expected):
    assert get_file(path_input) == expected

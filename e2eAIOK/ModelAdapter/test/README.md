# Transfer Learning Test

## Usage

1. Define your test file in `test/`.
2. If you want mannual test, run `python run_testting.py`. And the test has been integrated with Jenkins, so no extra work is needed.  

`run_testting.py` will collects all test files under `test/` and perform pytest.

## Test File

1. To integrate with `pytest`, you should use `import pytest`  in your test file.
2. To solve module dependency, you should insert many path to `sys.path` in head of your test file like this:
   ```commandline
   #!/usr/bin/python
   import pytest
   import sys
   import os
   sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),"src"))
   sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),"test"))
   ```
   The test file will correctly find any symbol under `src/` and `test/`



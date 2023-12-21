"""
 Copyright 2024 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

from transformers import Pipeline
from .utils import PIIEntityType


def detect_name_password(content: str, pipeline: Pipeline, entity_types=None):
    """Detects name and password in a string using bigcode/starpii model
    Args:
      entity_types: detection types
      pipeline: a transformer model
      content (str): A string containing the text to be analyzed.
    Returns:
        A list of dicts containing the tag type, the matched string, and the start and
        end indices of the match.
    """
    if entity_types is None:
        entity_types = [PIIEntityType.NAME, PIIEntityType.PASSWORD]
    matches = []
    try:
        for entity in pipeline(content):
            entity_group = entity["entity_group"]
            if ("NAME" == entity_group and PIIEntityType.NAME in entity_types) or \
                    ("PASSWORD" == entity_group and PIIEntityType.PASSWORD in entity_types):
                matches.append(
                    {
                        "tag": entity_group,
                        "value": entity["word"],
                        "start": entity["start"],
                        "end": entity["end"],
                    }
                )
    except:
        pass

    return matches

from pyrecdp.models.model_utils import get_pipeline
from loguru import logger



def detect_name_password(content):
    """Detects name and password in a string using bigcode/starpii model
    Args:
      content (str): A string containing the text to be analyzed.
      sentences ([str]): A string array
    Returns:
        A list of dicts containing the tag type, the matched string, and the start and
        end indices of the match.
    """
    matches = []
    pipe = get_pipeline(model_key='bigcode/starpii', task_key='token-classification', grouped_entities=True)
    try:
        for entity in pipe(content):
            entity_group = entity["entity_group"]
            if "NAME" == entity_group or "PASSWORD" == entity_group:
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

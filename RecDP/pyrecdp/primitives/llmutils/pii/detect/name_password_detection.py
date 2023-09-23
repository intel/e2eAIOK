from transformers import Pipeline


def detect_name_password(content: str, pipeline: Pipeline):
    """Detects name and password in a string using bigcode/starpii model
    Args:
      pipe: a transformer model
      content (str): A string containing the text to be analyzed.
      sentences ([str]): A string array
    Returns:
        A list of dicts containing the tag type, the matched string, and the start and
        end indices of the match.
    """
    matches = []
    try:
        for entity in pipeline(content):
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

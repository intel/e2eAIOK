from transformers import pipeline, Pipeline


def detect_name_password(content: str, context) -> []:
    """Detects name and password in a string using bigcode/starpii model
    Args:
      context: pii detect context
      content (str): A string containing the text to be analyzed.
    Returns:
        A list of dicts containing the tag type, the matched string, and the start and
        end indices of the match.
    """
    try:
        pipe = context["pipelines"]["name_password"]
    except KeyError:
        pipe = pipeline("token-classification", model="bigcode/starpii",
                        grouped_entities=True)

    matches = []
    for entity in pipe(content):
        entity_group = entity["entity_group"]
        if "EMAIL" == entity_group:
            continue

        matches.append(
            {
                "tag": entity_group,
                "value": entity["word"],
                "start": entity["start"],
                "end": entity["end"],
            }
        )
    return matches

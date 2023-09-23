import phonenumbers


def detect_phones(text):
    """Detects phone number in a string using phonenumbers libray
    Args:
      text (str): A string containing the text to be analyzed.
    Returns:
        A list of dicts containing the tag type, the matched string, and the start and
        end indices of the match.
    """
    matches = []

    for match in phonenumbers.PhoneNumberMatcher(text, "IN"):
        matches.append(
            {
                "tag": "PHONE_NUMBER",
                "value": match.raw_string,
                "start": match.start,
                "end": match.end,
            }
        )
    return matches

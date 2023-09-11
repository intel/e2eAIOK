import phonenumbers


def detect_phones(content):
    """Detects phone in a string using phonenumbers libray only detection the international phone number"""
    matches = []

    for match in phonenumbers.PhoneNumberMatcher(content, "IN"):
        matches.append(
            {
                "tag": "PHONE_NUMBER",
                "value": match.raw_string,
                "start": match.start,
                "end": match.end,
            }
        )
    return matches

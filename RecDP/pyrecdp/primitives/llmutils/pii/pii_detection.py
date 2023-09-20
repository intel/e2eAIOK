from .detect.emails_detection import detect_email
from .detect.ip_detection import detect_ip
from .detect.name_passwordy_detection import detect_name_password
from .detect.phones_detection import detect_phones


def scan_pii_text(text, context=None):
    """Detects ip,email,phone,name and password in a string
    Args:
      context (dict): pii detect context
      text (str): A string containing the text to be analyzed.
    Returns:
        A list of dicts containing the tag type, the matched string, and the start and
        end indices of the match.
    """
    result = []
    # use a regex to detect ip addresses
    result = result + detect_ip(text, context)
    # use a regex to detect emails
    result = result + detect_email(text, context)
    # for phone number use phonenumbers tool
    result = result + detect_phones(text, context)

    # for name and password use huggingface ner model
    result = result + detect_name_password(text, context)
    return result

from .detect.ip_detection import detect_ip
from .detect.emails_detection import detect_email
from .detect.phones_detection import detect_phones


def scan_pii_text(text):
    result = []
    # use a regex to detect ip addresses
    result = result + detect_ip(text)
    # use a regex to detect emails
    result = result + detect_email(text)
    # for phone number use phonenumbers tool
    result = result + detect_phones(text)
    return result

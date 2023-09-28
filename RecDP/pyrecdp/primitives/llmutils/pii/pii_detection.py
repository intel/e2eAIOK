from .detect.ip_detection import detect_ip
from .detect.emails_detection import detect_email
from .detect.phones_detection import detect_phones
from .detect.name_password_detection import detect_name_password
from transformers import Pipeline


def scan_pii_text(text:str, pipeline: Pipeline):
    result = []
    # use a regex to detect ip addresses
    result = result + detect_ip(text)
    # use a regex to detect emails
    result = result + detect_email(text)
    # for phone number use phonenumbers tool
    result = result + detect_phones(text)

    # for phone number use phonenumbers tool
    result = result + detect_name_password(text, pipeline)
    return result

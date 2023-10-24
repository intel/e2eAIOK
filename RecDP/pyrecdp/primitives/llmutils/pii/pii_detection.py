from .detect.ip_detection import detect_ip
from .detect.emails_detection import detect_email
from .detect.phones_detection import detect_phones
from .detect.name_password_detection import detect_name_password
from .detect.keys_detection import detect_keys
from .detect.utils import PIIEntityType

from transformers import Pipeline
from typing import List


def scan_pii_text(text: str, pipeline: Pipeline, entity_types: List[PIIEntityType] = None):
    result = []
    # use a regex to detect ip addresses

    if entity_types is None:
        entity_types = PIIEntityType.default()

    if PIIEntityType.IP_ADDRESS in entity_types:
        result = result + detect_ip(text)
    # use a regex to detect emails
    if PIIEntityType.EMAIL in entity_types:
        result = result + detect_email(text)
    # for phone number use phonenumbers tool
    if PIIEntityType.PHONE_NUMBER in entity_types:
        result = result + detect_phones(text)
    if PIIEntityType.KEY in entity_types:
        result = result + detect_keys(text)

    if PIIEntityType.NAME in entity_types or PIIEntityType.PASSWORD in entity_types:
        result = result + detect_name_password(text, pipeline, entity_types)
    return result

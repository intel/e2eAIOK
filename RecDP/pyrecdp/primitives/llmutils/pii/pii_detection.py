import json

from .utils.emails_ip_addresses_detection import detect_email_addresses
from .utils.phones_detection import detect_phones


def postprocess_secrets(secrets):
    """Postprocess the secrets found by the scan_secrets function"""
    if secrets:
        matches = json.dumps(secrets)
        has_secrets = True
    else:
        matches = json.dumps([])
        has_secrets = False
    return matches, has_secrets


def scan_pii_batch(examples):
    """Scan a batch of examples from a dataset to detect PII
    This add two columns to the dataset:
    - secrets: (list) of secrets/PII found
    - has_secrets: (bool) whether the example contains secrets/PII
    """
    list_secrets = []
    list_has_secrets = []
    number_secrets = []
    for text in examples["content"]:
        secrets = []
        # use a regex to detect keys + emails + ips
        secrets = secrets + detect_email_addresses(
            text, tag_types={"KEY", "EMAIL", "IP_ADDRESS"}
        )
        # detect phone number
        secrets = secrets + detect_phones(text)

        # to add this as new columns to datasets we need the same number of samples in each row
        # we save secrets as json strings instead of lists
        matches, has_secrets = postprocess_secrets(secrets)
        list_secrets.append(matches)
        list_has_secrets.append(has_secrets)
        number_secrets.append(len(secrets))
    return {
        "secrets": list_secrets,
        "has_secrets": list_has_secrets,
        "number_secrets": number_secrets,
    }

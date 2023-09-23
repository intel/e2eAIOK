from typing import Any, Dict
import numpy as np

from .operator import OPERATORS, Operator

from .pii.pii_redaction import redact_pii_text, random_replacements
from .pii.pii_detection import scan_pii_text


@OPERATORS.register_module("pii_remove")
class PiiRemove(Operator):

    def __init__(self, new_text_column="text", secrets_column='secrets', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._replacements = random_replacements()
        self._new_text_column = new_text_column
        self._secrets_column = secrets_column

    def processRow(self, row: Dict[str, Any]) -> Dict[str, Any]:
        secrets = scan_pii_text(row[self.text_key])
        text, modified = redact_pii_text(row[self.text_key], secrets, self._replacements)
        row[self._new_text_column] = text
        row[self._secrets_column] = str(secrets)
        return row

    def processBatch(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        list_secrets = []
        list_text = []
        for text in batch[self.text_key]:
            secrets = scan_pii_text(text)
            text, modified = redact_pii_text(text, secrets, self._replacements)
            list_secrets.append(str(secrets))
            list_text.append(text)

        batch[self._new_text_column] = np.array(list_text)
        batch[self._secrets_column] = np.array(list_secrets)
        return batch

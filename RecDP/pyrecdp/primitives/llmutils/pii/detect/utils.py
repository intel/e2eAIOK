from enum import Enum, auto


class PIIEntityType(Enum):
    IP_ADDRESS = auto()
    NAME = auto()
    EMAIL = auto()
    PHONE_NUMBER = auto()
    PASSWORD = auto()

    @classmethod
    def all(cls):
        return [member for name, member in PIIEntityType.__members__.items()]

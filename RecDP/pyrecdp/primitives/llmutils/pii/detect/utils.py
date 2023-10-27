from enum import Enum, auto


class PIIEntityType(Enum):
    IP_ADDRESS = auto()
    NAME = auto()
    EMAIL = auto()
    PHONE_NUMBER = auto()
    PASSWORD = auto()
    KEY = auto()

    @classmethod
    def default(cls):
        return [PIIEntityType.IP_ADDRESS, PIIEntityType.EMAIL, PIIEntityType.PHONE_NUMBER, PIIEntityType.KEY]

    @classmethod
    def parse(cls, entity):
        if "name" == entity:
            return PIIEntityType.NAME
        elif "password" == entity:
            return PIIEntityType.PASSWORD
        elif "email" == entity:
            return PIIEntityType.EMAIL
        elif "phone_number" == entity:
            return PIIEntityType.PHONE_NUMBER
        elif "ip" == entity:
            return PIIEntityType.IP_ADDRESS
        elif "key" == entity:
            return PIIEntityType.KEY
        else:
            raise NotImplementedError(f" entity type {entity} is not supported!")

    def getValue(self):
        if self == PIIEntityType.NAME:
            return "name"
        elif self == PIIEntityType.PASSWORD:
            return "password"
        elif self == PIIEntityType.EMAIL:
            return "email"
        elif self == PIIEntityType.PHONE_NUMBER:
            return "phone_number"
        elif self == PIIEntityType.IP_ADDRESS:
            return "ip"
        else:
            return "key"

"""
 Copyright 2024 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

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
            return PIIEntityType.PHONE_NUMBER
        elif "key" == entity:
            return PIIEntityType.KEY
        else:
            raise NotImplementedError(f" entity type {entity} is not supported!")

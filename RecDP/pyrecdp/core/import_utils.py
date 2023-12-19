import os
from typing import Optional
import pip
import importlib
import pathlib
import pkg_resources
from loguru import logger
import ray

FORCE_INSTALL=True

def list_requirements(requirements_path):
    with pathlib.Path(requirements_path).open() as requirements_txt:
        install_requires = [
            str(requirement)
            for requirement
            in pkg_resources.parse_requirements(requirements_txt)
        ]
    return install_requires

def fix_package_name(package):
    a = package.split('>')[0]
    a = a.split('<')[0]
    b = a.split('=')[0]
    
    package_name_map = {
        'scikit-learn' : 'sklearn',
        'pyyaml' : 'yaml',
        'Faker' : 'faker',
        'python-docx': 'docx',
        'Pillow': 'pillow',
        'alt-profanity-check': 'profanity_check',
        'faiss-cpu': 'faiss',
        'faiss-gpu': 'faiss',
        'python-docx': 'docx',
        'openai-whisper': 'whisper',
    }
    
    if b in package_name_map:
        b = package_name_map[b]
    #print(b)
    return b

class SessionENV:
    pip_list=[]
    system_list=[]
    @classmethod
    def update_pip_list(cls, package_or_list):
        def actual_func(package):
            if package not in cls.pip_list:
                cls.pip_list.append(package)
        if isinstance(package_or_list, list):
            for package in package_or_list:
                actual_func(package)
        elif isinstance(package_or_list, str):
            actual_func(package_or_list)
        else:
            raise ValueError(f"{package_or_list} with type of {type(package_or_list)} is not supported.")
        
    @classmethod
    def get_pip_list(cls):
        return cls.pip_list

    @classmethod
    def update_system_list(cls, package_or_list):
        def actual_func(package):
            if package not in cls.pip_list:
                cls.system_list.append(package)
        if isinstance(package_or_list, list):
            for package in package_or_list:
                actual_func(package)
        elif isinstance(package_or_list, str):
            actual_func(package_or_list)
        else:
            raise ValueError(f"{package_or_list} with type of {type(package_or_list)} is not supported.")

    @classmethod
    def get_system_list(cls):
        return cls.system_list

    @classmethod
    def clean(cls):
        cls.pip_list = []
        cls.system_list = []

def pip_install(package_or_list, verbose = 0, force = True):
    if package_or_list == "" or package_or_list == []:
        return
    SessionENV.update_pip_list(package_or_list)
    if verbose == 1:
        logger.info(f"SessionENV pip list is {SessionENV.get_pip_list()}")
    if force:
        local_pip_install(SessionENV.get_pip_list())

def system_install(package_or_list, verbose = 1, force = True):
    if package_or_list == "" or package_or_list == []:
        return
    SessionENV.update_system_list(package_or_list)
    if verbose == 1:
        logger.info(f"SessionENV pip list is {SessionENV.get_system_list()}")
    if force:
        local_system_install(SessionENV.get_system_list())
                
def local_pip_install(package_or_list):
    logger.info(f"local_pip_install for {package_or_list}")
    for package in package_or_list:
        try:                
            pip_name = fix_package_name(package)
            importlib.import_module(pip_name)
        except:
            pip.main(['install', '-q', package])
            
def local_system_install(package_or_list):
    logger.info(f"local_system_install for {package_or_list}")
    for package in package_or_list:
        os.system(f"apt install -y {package}")
 


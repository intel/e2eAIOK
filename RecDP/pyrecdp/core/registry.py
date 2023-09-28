class Registry(object):
    def __init__(self, registry_name):
        self._name = registry_name
        self._modules = {}
       
    @property 
    def modules(self):
        return self._modules
    
    def register(self, cls):
        if cls is not None:
            name = cls.__name__
            self._modules[name] = cls
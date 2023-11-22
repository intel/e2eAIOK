class Registry(object):
    def __init__(self, registry_name):
        self._name = registry_name
        self._modules = {}
       
    @property 
    def modules(self):
        return self._modules
    
    def register(self, cls, name = None):
        if cls is not None:
            if name is None:
                name = cls.__name__
            self._modules[name] = cls

def new_instance(module, clazz, **clazz_kwargs):
    import importlib
    module = importlib.import_module(module)
    class_ = getattr(module, clazz)
    return class_(**clazz_kwargs)




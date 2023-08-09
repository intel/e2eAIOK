
def execute_asynchronously(func, *args, **kwargs):
    from threading import Thread

    t = Thread(target=func, args=args, kwargs=kwargs, daemon=True)
    t.start()
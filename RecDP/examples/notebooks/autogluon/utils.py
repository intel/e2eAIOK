import timeit

class Timer:
    level = 0

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = timeit.default_timer()
        Timer.level += 1

    def __exit__(self, *a, **kw):
        Timer.level -= 1
        print(
            f'{"  " * Timer.level}{self.name} took {timeit.default_timer() - self.start} sec')

import gc
import time

from contextlib import contextmanager


@contextmanager
def time_func(name):

    gc.disable()
    st = time.time()
    yield
    et = time.time()
    gc.enable()
    print('[{}] finished in {} ms'.format(name, int((et - st) * 1000)))

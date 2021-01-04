import time


def timer(func):
    def wrap(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)

        # See https://jax.readthedocs.io/en/latest/async_dispatch.html
        # for why this is needed.
        for return_item in ret:
            _apply_blocking(return_item)

        stop = time.time()
        duration = (stop - start) * 1000.0
        print("{:s} duration: {:.3f} ms".format(func.__name__, duration))
        return ret

    return wrap


def _apply_blocking(function_return_item):
    try:
        function_return_item.block_until_ready()
        return
    except AttributeError:
        pass

    for _, dict_item in function_return_item.items():
        try:
            dict_item.block_until_ready()
        except AttributeError:
            pass

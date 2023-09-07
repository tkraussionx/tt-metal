import types
from functools import wraps
import os
from loguru import logger
import time

functionsToWrap = []
functionsToWrapStr = os.getenv("TT_METAL_TT_LIB_FUNCT_REMOVE_LIST")
if functionsToWrapStr:
    tmp = functionsToWrapStr.strip("[]").split(",")
    for functionStr in tmp:
        functionsToWrap.append(functionStr.strip('" '))

stop = None
start = None

def decorator(f, name):
    @wraps(f)
    # Just wrapping C++ calls with a python side warpper is enough
    # This will let them be picked up by the python settrace
    def wrapper(*args, **kwargs):
        local_name = name
        if start:
            start()
        ret = f(*args, **kwargs)
        if stop:
            stop()
        return ret

    return wrapper

def tt_lib_funct_wrapper(module):
    # This function is called on the c++ side which dose not dump exceptions
    # Try catch is added to dump the exception to stdout
    try:
        for name in dir(module):
            obj = getattr(module, name)
            # TODO: Improve finding objects , __ search is a very bad idea
            if callable(obj) and "__" not in name:
                if name in functionsToWrap:
                    logger.info(f'Wrapped function "{name}"')
                    setattr(module, name, decorator(obj, name))
                tt_lib_funct_wrapper(obj)
    except e as Exception:
        print(e)
        raise e

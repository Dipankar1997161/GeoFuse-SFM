import logging
import time
from contextlib import contextmanager

def make_logger(name: str = "sfm", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.setLevel(level)
    return logger

@contextmanager
def timed(logger: logging.Logger, msg: str):
    t0 = time.time()
    logger.info(f"{msg} ...")
    yield
    dt = time.time() - t0
    logger.info(f"{msg} done in {dt:.2f}s")

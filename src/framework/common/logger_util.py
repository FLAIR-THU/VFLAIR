import logging
import os

CURRENT_PATH = os.path.dirname(
    os.path.dirname(__file__)
)
LOG_PATH = os.path.join(CURRENT_PATH, "logs")
os.makedirs(LOG_PATH, exist_ok=True)


def get_logger(name="root"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler("{0}/{1}.log".format(LOG_PATH, name), mode='a')
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(file_handler)
    return logger

import logging
import sys
import logging


def enable_logging():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
    logger = logging.getLogger()
    logger.disabled = False
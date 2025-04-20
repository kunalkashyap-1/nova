import logging
import sys

logger = logging.getLogger("nova")

formatter = logging.Formatter(
    fmt = "%(asctime)s - %(levelname)s -  %(name)s - %(message)s"
)

stream_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler("nova.log")

stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.handlers = [stream_handler, file_handler]

logger.setLevel(logging.INFO)

logger.info("Logger initialized")
from transformers.utils import logging

#logging.set_verbosity_info()   # 👈 关键
#logging.enable_default_handler()
#print(logging.get_verbosity())  # 应该变成 20
logger = logging.get_logger(__name__)

for i in range(5):
    logger.warning_once("Loading model...")

logger.warning_once("sb")

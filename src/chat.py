import os
from utils import setup_logger
from pathlib import Path


# Config
logger = setup_logger(__name__)
cwd = Path(os.getcwd())


def chat(input, model, *args, **kwargs):
    logger.info("Chat function called")
    return f"This is a placeholder for the chat function. Model chosen: {model}. Input: {input}"

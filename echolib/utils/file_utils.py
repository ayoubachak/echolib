import os
from typing import Optional
from common.logger import logger

def read_file(file_path: str) -> Optional[str]:
    if not os.path.exists(file_path):
        logger.error(f"File {file_path} does not exist.")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            logger.debug(f"Read content from {file_path}")
            return content
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}")
        return None

def write_file(file_path: str, content: str) -> bool:
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
            logger.debug(f"Wrote content to {file_path}")
            return True
    except Exception as e:
        logger.error(f"Failed to write to {file_path}: {e}")
        return False

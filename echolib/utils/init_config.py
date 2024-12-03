import os
from pathlib import Path
from echolib.common.logger import logger
from echolib.utils.file_utils import read_file, write_file

def initialize_config(config_dir: str = None):
    from echolib.common.globals import Globals
    if config_dir is None:
        config_dir = Globals().config_dir
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)

    templates_dir = Path(__file__).parent.parent / 'templates'
    for template in templates_dir.glob("*.json.template"):
        target_file = config_dir / template.stem.replace('.template', '')  # Remove .template
        if not target_file.exists():
            content = read_file(str(template))
            if content:
                success = write_file(str(target_file), content)
                if success:
                    logger.info(f"Created configuration file: {target_file}")
                else:
                    logger.error(f"Failed to create configuration file: {target_file}")
        else:
            logger.warning(f"Configuration file already exists: {target_file}")

if __name__ == "__main__":
    initialize_config()

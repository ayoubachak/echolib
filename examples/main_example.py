from echolib import ai_models
from echolib.common.logger import logger
from echolib.models.base import BaseModel
from time import time
from pathlib import Path

def main():
    model: BaseModel = list(ai_models.models.values())[-1]  # Assuming LMStudioModel is last
    result = model.sys_inference(
        sys_prompt="You are an intelligent assistant, genius in geography and history, and concise.",
        usr_prompt="What's the independence date of Morocco?",
        seed=int(time())
    )
    logger.info(f"Result: {result}")
    print(result)

if __name__ == "__main__":
    main()

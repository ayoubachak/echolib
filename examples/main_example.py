from echolib import ai_models
from echolib.common.logger import logger
from time import time

from echolib.models.base import BaseModel

def main():
    model : BaseModel = list(ai_models.models.values())[-1]  # Assuming LMStudioModel is last
    result = model.sys_inference(
        sys_prompt="You are an intelligent assistant, genius in geography and history, and concise.",
        usr_prompt="What's the independence date of Morocco?",
        seed=int(time())
    )
    logger.info(f"Result: {result}")

if __name__ == "__main__":
    main()

# examples/main_example.py

from echolib import model_manager
from echolib.common.logger import logger
from echolib.models.base import BaseModel
from time import time

def main():
    model: BaseModel = model_manager.get_model("LM Studio")
    assert model is not None, "LM Studio model not loaded."
    result = model.sys_inference(
        sys_prompt="You are an intelligent assistant, genius in geography and history, and concise.",
        usr_prompt="What's the independence date of Morocco?",
        seed=int(time())
    )
    logger.info(f"Result: {result}")
    print(result)

if __name__ == "__main__":
    main()

from models import ai_models
from common import logger
from time import time

from models import HuggingFaceModel


for name, instance in ai_models.models.items():
    print(name, instance)

second_model : HuggingFaceModel = ai_models.models[list(ai_models.models.keys())[-1]]
# second_model.set_max_itterations(1)
result = second_model.sys_inference(
    "You are an intelligent assistant, genius in geography, and history, and you don't yapp.",
    "What's the independance date of Morocco ?",
    seed=int(time())
    )

logger.info(f"Result: {result}")



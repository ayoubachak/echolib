from models import ai_models
from common import logger
from time import time

from models import HuggingFaceModel
import os

# Assert the existence of resume.txt
assert os.path.exists('resume.txt'), "resume.txt does not exist"

# Assert the existence of job.txt
assert os.path.exists('job.txt'), "job.txt does not exist"

for name, instance in ai_models.models.items():
    print(name, instance)

second_model : HuggingFaceModel = ai_models.models[list(ai_models.models.keys())[1]]
# second_model.set_max_itterations(1)
with open('resume.txt', 'r') as file:
    resume = file.read()
with open('job.txt', 'r') as file:
    job = file.read()
result = second_model.sys_inference(
    "You are an intelligent assistant, genius in software engineering and know what HR managers like to hear, for this matter you write great cover letters given only the resume and the job posting.",
    "Here is my resume : \n" + resume + "\n" + "Here is the job posting : \n" + job       
    ,
    seed=1720529636 #int(time())
    )

with open('cover_letter.txt', 'w',encoding="utf-8") as file:
    file.write(result)

logger.info(f"Result: {result}")
logger.info("Inference with tokens: "+second_model.inference_with_tokens('What\'s the capital of France?', max_new_tokens=2))



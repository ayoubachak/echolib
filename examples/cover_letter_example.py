from echolib import ai_models
from echolib.models.base import BaseModel
from echolib.utils.file_utils import read_file, write_file
from time import time

def generate_cover_letter(resume_path: str, job_path: str, output_path: str = "cover_letter.txt") -> None:
    resume = read_file(resume_path)
    job = read_file(job_path)
    if resume is None or job is None:
        print("Missing resume or job description.")
        return

    model : BaseModel = list(ai_models.models.values())[0]  # Assuming HuggingFaceModel is first
    result = model.sys_inference(
        sys_prompt="You are an intelligent assistant, genius in software engineering and know what HR managers like to hear. You write great cover letters given only the resume and the job posting.",
        usr_prompt=f"Here is my resume:\n{resume}\nHere is the job posting:\n{job}",
        seed=int(time())
    )

    success = write_file(output_path, result)
    if success:
        print(f"Cover letter generated and saved to {output_path}.")
    else:
        print("Failed to write the cover letter.")

if __name__ == "__main__":
    generate_cover_letter("resume.txt", "job.txt")

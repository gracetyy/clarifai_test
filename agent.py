import os
from dotenv import load_dotenv
from clarifai.client.user import User
from clarifai.client import Model
from crewai import Agent, Task, Crew, Process, LLM
from openai import OpenAI

load_dotenv()
CLARIFAI_PAT = os.getenv('CLARIFAI_PAT')
CLARIFAI_USER_ID = os.getenv('CLARIFAI_USER_ID')
if CLARIFAI_PAT is None:
    raise ValueError("CLARIFAI_PAT is not set. Please set the CLARIFAI_PAT environment variable.")

# Configure Clarifai LLM
clarifai_llm = Model(
    url="https://clarifai.com/deepseek-ai/deepseek-chat/models/DeepSeek-R1-0528-Qwen3-8B"
)

if __name__ == "__main__":
    user_input = input("Enter the topic to research: ").strip()

    response = clarifai_llm.predict(prompt=user_input)
    print(response)
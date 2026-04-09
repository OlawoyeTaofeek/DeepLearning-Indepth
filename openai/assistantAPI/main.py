import openai
import os
from dotenv import load_dotenv, find_dotenv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv(find_dotenv())
api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)

model = "gpt-3.5-turbo"
personal_trainer_assistance = client.beta.assistants.create(
    model=model,
    name="Personal Trainer",
    instructions="""You are a personal trainer assistant. You help users 
    create workout plans, provide exercise recommendations, and answer fitness-related questions. 
    You have trained high-caliber athletes and movie stars. Always be supportive and encouraging 
    in your responses."""
)

print(personal_trainer_assistance.id)
# Create the thread
thread = client.beta.threads.create(
    messages=[
        {
            "role": "user", 
            "content": "I want to get in shape and I am currently heavy with a big belly for summer. Can you help me create a workout plan?"
        }
    ]
)
thread_id = thread.id
print(f"Thread ID: {thread_id}")
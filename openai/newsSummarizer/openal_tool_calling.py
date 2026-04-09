import json

from utils import Utils
from openai import OpenAI
import requests
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
from typing import List
import time   
from datetime import datetime


open_ai_api_key, news_api_key = Utils.get_api_key()
model = ...

class OpenAIToolCalling:
    """Class to handle OpenAI tool calling."""
    thread_id = None 
    assistant_id = None

    def __init__(self, model: str = model):
        self.client = OpenAI(api_key = open_ai_api_key)
        self.model = model 
        self.assistant = None
        self.thread = None
        self.run = None  
        self.summary = None  

        if OpenAIToolCalling.assistant_id:
            self.assistant = self.client.beta.assistants.retrieve(
                assistant_id=OpenAIToolCalling.assistant_id
            )
        if OpenAIToolCalling.thread_id:
            self.thread = self.client.beta.threads.retrieve(
                thread_id=OpenAIToolCalling.thread_id
            )

    def get_summary(self):
        return self.summary

    def creete_assistance(self, name, instructions: str, tools: List):
        if not self.assistant:
            assistant_obj = self.client.beta.assistants.create(
                name=name,
                instructions=instructions,
                tools=tools,
                model = self.model
            )

            OpenAIToolCalling.assistant_id = assistant_obj.id 
            self.assistant = assistant_obj

    def create_thread(self):
        if not self.thread:
            thread_obj = self.client.beta.threads.create()
            OpenAIToolCalling.thread_id = thread_obj.id 
            self.thread = thread_obj

    def add_message_to_thread(self, role, content):
        if self.thread:
           self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role=role, 
                content=content
            )
           
    def run_assistance(self, instructions):
        if self.thread and self.assistant:
            self.run = self.client.beta.threads.runs.create(
                thread_id = self.thread.id,
                assistant_id=self.assistant.id,
                instructions=instructions
            )

    def process_messages(self):
        if self.thread:
            messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
            summary = []

            last_messages = messages.data[0]
            role = last_messages.role
            response = last_messages.content[0].text.value
            summary.append(response)
            
            self.summary = "\n".join(summary)
            print(f"SUMMARY------> {role.capitalize()}: ==> {response}")
    
    def wait_for_completion(self):
        if self.thread and self.run:
            while True:
                time.sleep(5)
                run_status = self.client.beta.threads.runs.retrieve(
                    thread_id=self.thread.id 
                )

                print(f"RUN STATUS:: {run_status.model_dump_json(indent=4)}")

                if run_status.status == "completed":
                    self.process_messages()
                    break 
                elif run_status.status == "requires_action":
                    print("FUNCTION CALLING NOW")
                    self.call_required_functions(
                        required_actions = run_status.required_action.submit_tool_outputs.model_dump()
                    )
    
    def run_steps(self):
        run_steps = self.client.beta.threads.runs.steps.list(
            thread_id=self.thread.id,
            run_id=self.run.id
        )
        print(f"Run-steps:  {run_steps}")
        return run_steps
    
    def call_required_functions(self, required_actions):
        if not self.run:
            return 
        tool_outputs = []
        for action in required_actions['tool_calls']:
            func_name = action["function"]['name']
            arguments = json.loads(action['function']['arguments'])

            if func_name == "get_news":
                output = get_news(topic=arguments['topic'])
                print(f"THE OUTPUT:::::::{output}")
                
                final_str = ""
                for item in output:
                    final_str += "".join(item)
                tool_outputs.append({
                    "tool_call_id": action['id'],
                    "output": final_str
                })
            else:
                raise ValueError(f"Unknown function: {func_name}")
            print("Submitting back to the Assistant")
            self.client.beta.threads.runs.submit_tool_outputs(
                thread_id=self.thread.id,
                run_id=self.run.id,
                tool_outputs=tool_outputs
            )
        return tool_outputs

    
def get_news(topic: str) -> str:
    """Fetch news articles related to the given topic."""
    # Placeholder for news fetching logic using news_api_key
    url = (f"""https://newsapi.org/v2/everything?q={topic}
           &sortBy=popularity&apiKey={news_api_key}&pageSize=5""")

    try:
        response = requests.get(url)
        if response.status_code == 200:
            new = json.dumps(response.json(), indent=4)
            news_dict = json.loads(new)

            # Access all the fields in the news_dict as needed
            status = news_dict.get("status")
            total_results = news_dict.get("totalResults", 0)
            articles = news_dict.get("articles", [])
            print(f"The status was: {status} with {total_results} news returned")
            final_news = []

            for article in articles:
                source_name = article['source']['name']
                author = article['author']
                title = article['title']
                description = article['description']
                url = article['url']
                content = article['content'] 
                title_description = f"""
                   Title: {title},
                   Author: {author},
                   Source: {source_name}
                   Description: {description}
                   URL: {url}
                """
                final_news.append(title_description)
            return final_news
        else:
            return []

    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")



def main():
    news = get_news("bitcoin")
    print(news[0])

if __name__ == "__main__":
    main()


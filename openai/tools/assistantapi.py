# Tool/Function Calling has 5 steps 

from dotenv import load_dotenv, find_dotenv
import os
from openai import OpenAI
import time
import requests
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
_ = load_dotenv(find_dotenv())
import streamlit as st 

# Using the assistant api

class AssistantManager:
    thread_id = "thread_6um3pifj7prjzc87NLQSOChM"
    assistant_id = "asst_TsA8aS4sbtQym3yPfFYZoDjr"  

    def __init__(self, model: str):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.assistant = None 
        self.thread = None  
        self.run = None
        self.model = model 
        self.summary = None  

        if AssistantManager.assistant_id:
            self.assistant = self.client.beta.assistants.retrieve(
                assistant_id=AssistantManager.assistant_id
            )

        if AssistantManager.thread_id:
            self.thread = self.client.beta.threads.retrieve(
                thread_id=AssistantManager.thread_id
            )

    def create_assistant(self, name: str, instructions: str, tools: list):
        if not self.assistant:
            assistant_obj = self.client.beta.assistants.create(
                name=name, 
                instructions=instructions, 
                model=self.model,
                tools=tools
            )

            AssistantManager.assistant_id = assistant_obj.id
            self.assistant = assistant_obj

    def create_thread(self):
        if not self.thread:
            thread_obj = self.client.beta.threads.create()
            AssistantManager.thread_id = thread_obj.id 
            self.thread = thread_obj


    def create_message(self, content: str, role: str):
        if self.thread:
            message = self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role=role,
                content=content
            )
            return message

    def run_assistant(self, instructions: str):
        if self.thread and self.assistant:
            self.run = self.client.beta.threads.runs.create(
                thread_id=self.thread.id,
                assistant_id=self.assistant.id,
                instructions=instructions
            )
        return self.run
    
    def get_summary(self):
        return self.summary
    
    def run_steps(self):
        run_steps = self.client.beta.threads.runs.steps.list(
            thread_id=self.thread.id,
            run_id=self.run.id
        )

        steps_data = []

        for step in run_steps.data:
            steps_data.append(step.model_dump())

        return json.dumps(steps_data, indent=4)

    def wait_for_completion(self):
        if self.thread and self.run:
            while True:
                time.sleep(5)
                run_status = self.client.beta.threads.runs.retrieve(
                    thread_id=self.thread.id,
                    run_id=self.run.id
                )

                print(f"RUN STATUS:: {run_status.model_dump_json(indent=4)}")

                if run_status.status == "completed":
                    self.process_messages()
                    break 
                elif run_status.status == "requires_action":
                    print("FUNCTION CALLING NOW")
                    self.call_required_functions(
                        run_status = run_status
                    )
                    print(f"The output from run_status is: \n{run_status}")
            
    def call_required_functions(self, run_status):
        tool_calls = run_status.required_action.submit_tool_outputs.tool_calls

        tool_outputs = []
        for tool in tool_calls:
            function_name = tool.function.name
            arguments = json.loads(tool.function.arguments)

            print("FUNCTION NAME:", function_name)
            print("ARGUMENTS:", arguments)

            if function_name == "get_news":
                topic = arguments["topic"]
                news = get_news(topic)
                final_str = "\n\n".join(news)

                tool_outputs.append(
                    {
                        "tool_call_id": tool.id,
                        "output": final_str
                    }
                )
            else:
                raise ValueError(f"Unknown function: {function_name}")
            print("Submitting back to the Assistant")
        self.client.beta.threads.runs.submit_tool_outputs(
            thread_id=self.thread.id,
            run_id=self.run.id,
            tool_outputs=tool_outputs
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


import json 

news_api_key = os.getenv("NEWS_API_KEY")
def get_news(topic: str) -> str:
    """Fetch news articles related to the given topic."""
    # Placeholder for news fetching logic using news_api_key
    url = (f"""https://newsapi.org/v2/everything?q={topic}
           &apiKey={news_api_key}&pageSize=5""")

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


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Get news sorted by popularity",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic to search news for, e.g. Isrel War with Iran",
                    }
                },
                "required": ["topic"]
            }
        }
    }
]

def main():
    assistantmanager = AssistantManager(model="gpt-4.1-mini")

    ## Create a streamlit UI interface
    st.title("New Summarizer")
    with st.form(
        key="user_input_form"):
        instructions = st.text_input("Enter topic:")
        submit_button = st.form_submit_button(label="Run Assistant")
    
        if submit_button:
            assistantmanager.create_assistant(
                name="News Summarizer",
                instructions="""
                You are a professional news assistant.

                Your job is to fetch news articles and format them clearly.

                Follow this exact structure:

                ------------------------

                NEWS ARTICLES

                Article 1
                Title:
                Author:
                Description:
                URL:

                Article 2
                Title:
                Author:
                Description:
                URL:

                Article 3
                Title:
                Author:
                Description:
                URL:

                ------------------------

                FINAL SUMMARY

                Write a clear short summary of all the news in 2-5 lines.

                ------------------------

                Rules:

                1. Each article must be clearly separated
                2. Title, Author, Description, and URL must be on separate lines
                3. Add spacing between articles
                4. After listing all articles, write the summary
                5. Never merge everything into one paragraph
                6. Always keep clean readable formatting
                """,
                tools=tools
            )

            assistantmanager.create_thread()

            assistantmanager.create_message(
                content=f"Summarize the news on this topic {instructions}",
                role="user"
            )

            assistantmanager.run_assistant(
                instructions="""
                    Get the news using the tool.
                    Then display:

                    1. Title
                    2. Author
                    3. Description
                    4. URL

                    for each article. ensure the are property seperated and also they should be line by line

                    After listing all articles,
                    provide a final short summary of the news.
                """
            )
            # wait for completiona and process messages
            assistantmanager.wait_for_completion()
            summary = assistantmanager.get_summary()
            st.markdown(summary)

            st.text("Run Steps:")
            st.code(assistantmanager.run_steps(), line_numbers=True)


if __name__ == "__main__":
    main()


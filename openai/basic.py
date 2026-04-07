from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import os 
from typing import List, Union
from Utils import Utils


class OpenAIClient:
    def __init__(self):
        self.api_key = Utils.load_api_key()
        self.client = OpenAI(api_key=self.api_key)

    # Using old legacy method for compatibility with older OpenAI API versions
    def create_chat_completion(self, model: str, messages: list, max_tokens: int = 150):
        """Create a chat completion using the OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print("Error creating chat completion:", e)
            return None
        
    # Using new method for newer OpenAI API versions
    def create_completion(self, model: str, prompt: Union[str, List], max_tokens: int = 150):
        """Create a text completion using the OpenAI API."""
        try:
            response = self.client.responses.create(
                model=model,
                input=prompt,
                max_output_tokens=max_tokens
            )
            # print("Raw Completion Response:", response)
            return response
        except Exception as e:
            print("Error creating completion:", e)
            return None
    @staticmethod
    def output_response(self, response):
        """Return the output from the API response."""
        if response:
            return response
        else:
            print("No valid response to return.")
            return None
        
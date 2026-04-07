from dotenv import load_dotenv, find_dotenv
import os

class Utils:
    @staticmethod
    def load_api_key():
        """Load the OpenAI API key from environment variables."""
        _ = load_dotenv(find_dotenv())
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        return api_key
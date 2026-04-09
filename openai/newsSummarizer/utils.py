from dotenv import load_dotenv, find_dotenv

class Utils:
    """Load the OpenAI API key from environment variables."""
    
    @staticmethod
    def get_api_key():

        import os
        _ = load_dotenv(find_dotenv())
        api_key = os.getenv("OPENAI_API_KEY")
        news_api_key = os.getenv("NEWS_API_KEY")
        if not api_key and not news_api_key:
            raise ValueError("API keys not found in environment variables.")
        return api_key, news_api_key

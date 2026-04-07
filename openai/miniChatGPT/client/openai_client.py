from openai import OpenAI
from client.logger import setup_logger 

logger = setup_logger()

class OpenAIClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def stream_chat(self, messages, model="gpt-5-mini", max_tokens=400):
        """Stream chat responses from the OpenAI API."""
        logger.info("Streaming chat response.")
        try:
            with self.client.responses.stream(
                model=model,
                input=messages,
                max_output_tokens=max_tokens
            ) as stream:
                for event in stream:
                    if event.type == "response.output_text.delta":
                       yield event.delta
        except Exception as e:
            yield f"error: {e}"
            logger.error(f"Error during streaming: {e}")
            raise


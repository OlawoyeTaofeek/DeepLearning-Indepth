from openai import OpenAI

class OpenAIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)

    def chat_completion(self, model: str, messages: list, max_tokens: int = 300, stream: bool = False):
        """
        messages: list of dicts [{'role': 'user', 'content': '...'}, ...]
        stream: if True, returns a generator of tokens
        """
        try:
            if stream:
                with self.client.responses.stream(
                    model=model,
                    input=messages,
                    max_output_tokens=max_tokens
                ) as stream_response:
                    for chunk in stream_response:
                        if chunk.type == "response.output_text.delta":
                            yield chunk.delta
            else:
                response = self.client.responses.create(
                    model=model,
                    input=messages,
                    max_output_tokens=max_tokens
                )
                return response.output_text
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return None
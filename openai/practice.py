from openai import OpenAI
from Utils import Utils


class OpenAIPractice:
    """A simple wrapper for OpenAI responses API."""

    def __init__(self):
        self.api_key = Utils.load_api_key()
        self.client = OpenAI(api_key=self.api_key)

    def create_completion(
        self,
        model: str,
        topic: str,
        style: str,
        max_tokens: int = 150
    ) -> str:
        """
        Generate a completion for a topic in a given style.

        Args:
            model (str): Model name (e.g., 'gpt-4.1-mini').
            topic (str): The concept to explain.
            style (str): Explanation style.
            max_tokens (int): Maximum tokens in output.

        Returns:
            str: The text output from the model.
        """
        prompt = f"Explain the concept of {topic} in a {style}."

        try:
            # Use simple string input instead of nested dict
            response = self.client.responses.create(
                model=model,
                input=prompt,
                max_output_tokens=max_tokens
            )

            # Debug: print full response object if needed
            # print(response)

            # Return the text output
            return response.output_text

        except Exception as e:
            print("Error generating completion:", e)
            return None


def main():
    topics = [
        {"topic": "python loops", "style": "simple explanation with code examples"},
        {"topic": "vector embeddings", "style": "simple explanatory way"},
        {"topic": "sql joins", "style": "good explanation with SQL examples"}
    ]

    practice = OpenAIPractice()

    for item in topics:
        result = practice.create_completion(
            model="gpt-4.1-mini",  # reliable available model
            topic=item["topic"],
            style=item["style"],
            max_tokens=200
        )

        if result:
            print(f"\n📘 {item['topic'].upper()}")
            print(result)
            print("-" * 50)
        else:
            print(f"⚠️ No result for topic: {item['topic']}")


if __name__ == "__main__":
    main()
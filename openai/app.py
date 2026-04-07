from basic import OpenAIClient


client = OpenAIClient()
response = client.create_chat_completion(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    max_tokens=50
)

# print("Chat Completion Response:", response)

response = client.create_completion(
    model="gpt-5.4",
    prompt="Write a one-sentence bedtime story about a unicorn.",
    max_tokens=150
)

print(response)
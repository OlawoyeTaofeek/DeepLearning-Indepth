from client.openai_client import OpenAIClient
from client.utils import Utils
from memory.chat_memory import ChatMemory

def main(): 
    api_key = Utils.load_api_key()
    api_client = OpenAIClient(api_key=api_key)
    chat_memory = ChatMemory(max_memory_size=10)

    print("🤖 AI Chat Started. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting chat. Goodbye!")
            break
        
        # Add user message to memory
        chat_memory.add_message(role="user", content=user_input)

        # Get current conversation history
        messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
        messages += chat_memory.get_memory()

        # Get AI response (streaming)
        stream = api_client.chat_completion(
            model="gpt-4o",
            messages=messages,
            max_tokens=300,
            stream=True
        )

        ai_response = ""
        print("AI: ", end="", flush=True)
        for event in stream:
            ai_response += event
            print(event, end="", flush=True)
        print()  # Newline after response is complete

        # Add AI response to memory
        chat_memory.add_message(role="assistant", content=ai_response)

if __name__ == "__main__":
    main()
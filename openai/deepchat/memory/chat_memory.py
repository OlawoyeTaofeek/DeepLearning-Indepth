class ChatMemory:
    def __init__(self, max_memory_size: int = 10):
        # Store last n messages to provide context
        self.max_memory_size = max_memory_size
        self.memory = []
    
    def add_message(self, role: str, content: str):
        # limit memory history
        if len(self.memory) >= self.max_memory_size:
            self.memory.pop(0)  # Remove the oldest message
        self.memory.append({"role": role, "content": content})
    
    def get_memory(self):
        return self.memory.copy()
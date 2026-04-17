from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Literal, Union
import re

def extract_variables(template: str) -> List[str]:
    return re.findall(r"{(.*?)}", template)

class Message(BaseModel):
    role: Literal['system', 'user', 'assistant']
    content: str

    def to_openai(self) -> dict:
        return {"role": self.role, "content": self.content}
    
# schema.py
from pydantic import BaseModel, Field
from typing import List, Dict

class PromptTemplateSchema(BaseModel):
    input_variables: List[str] = Field(default_factory=list)
    input_types: Dict = Field(default_factory=dict)
    partial_variables: Dict = Field(default_factory=dict)
    template: str
    
class HumanMessage(BaseModel):
    ...

class HumanMessagePromptTemplate(BaseModel):
    ... 

class SystemMessage(BaseModel):
    ...

class SystemMessagePromptTemplate(BaseModel):
    ...

class ChatPromptTemplate:
    def __init__(self, messages: List[Union[Message, tuple]]):
        self.messages: List[Message] = []
        for msg in messages:
            if isinstance(msg, Message):
                self.messages.append(msg)
            elif isinstance(msg, tuple) and len(msg) == 2:
                role, content = msg
                self.messages.append(Message(role=role, content=content))
            else:
                raise ValueError("Each message must be a Message instance or a (role, content) tuple")

    def format_messages(self, **kwargs) -> List[Message]:
        formatted_messages = []
        for message in self.messages:
            formatted_content = message.content.format(**kwargs)
            formatted_messages.append(Message(role=message.role, content=formatted_content))
        return formatted_messages
    
    def from_template(template_string: str) -> 'ChatPromptTemplate':
        # This is a placeholder for a method that would parse a template string
        # and create a ChatPromptTemplate instance. The actual implementation would
        # depend on the specific template format you want to support.
        raise NotImplementedError("from_template method is not implemented yet")
    
    def from_messages(messages: List[Union[Message, tuple]]) -> 'ChatPromptTemplate':
        return ChatPromptTemplate(messages)
    
    @classmethod
    def messages(cls, messages: List[Union[Message, tuple]]):
        return cls(messages)
    
class ChatOpenAI:
    ...
    


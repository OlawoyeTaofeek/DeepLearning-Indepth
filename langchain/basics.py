from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

chat = ChatOpenAI(temperature=0.0)
print(chat)

template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""
template = PromptTemplate.from_template(template_string)
print(template.format(style="Shakespearean", text="What a wonderful day!"))
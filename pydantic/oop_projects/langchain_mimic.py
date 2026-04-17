# langchain_mimic.py
from schema import PromptTemplateSchema
import re


def extract_variables(template: str):
    return re.findall(r"{(.*?)}", template)


class PromptTemplate:
    def __init__(self, schema: PromptTemplateSchema):
        self.schema = schema

    @property
    def input_variables(self):
        return self.schema.input_variables 
    
    def __getattr__(self, name):
        return getattr(self.schema, name)
    
    def __repr__(self):
        return f"PromptTemplate(input_variables={self.input_variables}, input_types={self.schema.input_types}, partial_variables={self.schema.partial_variables}, template='{self.template}')"

    @classmethod
    def from_template(cls, template_string: str):
        variables = extract_variables(template_string)

        schema = PromptTemplateSchema(
            input_variables=variables,
            template=template_string
        )

        return cls(schema)

    def format(self, **kwargs) -> str:
        missing = set(self.schema.input_variables) - set(kwargs.keys())
        extra = set(kwargs.keys()) - set(self.schema.input_variables)

        if missing:
            raise ValueError(f"Missing variables: {missing}")
        if extra:
            raise ValueError(f"Unexpected variables: {extra}")

        return self.schema.template.format(**kwargs)

prompt = PromptTemplate.from_template("Explain {topic} in simple terms. using {language} language.")
print(prompt)
print(type(prompt))
print(prompt.input_variables)
print(prompt.template)

msgs = prompt.format(topic="AI", language="English")
print(msgs)
    

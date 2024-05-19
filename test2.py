from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# connecting model on huggingface
generator = pipeline("text-generation", model="gpt2")
llm = HuggingFacePipeline(pipeline=generator)

messages = [
    SystemMessage(content="please anwser all my words"),
    HumanMessage(content="hi! My name is Niem. Can I call you bot?"),
]
# using parser to print out llm content
response = llm.invoke(messages)
parser = StrOutputParser()
print(parser.invoke(response))

# using chain
chain = llm | parser
result = chain.invoke(messages)
print(chain.invoke(response))

# using prompt input
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
chain = prompt_template | llm | parser #chain will execute the left to right with the input as the left 
print(chain.invoke({"language": "italian", "text": "hi"}))
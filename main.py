from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import pipeline
# connecting model on huggingface
generator = pipeline("text-generation", model="gpt2")
llm = HuggingFacePipeline(pipeline=generator)

#creating prompt
template = "Write a poem about {topic}."
prompt = PromptTemplate(template=template, input_variables=["topic"])
chain = LLMChain(llm=llm, prompt=prompt)

poem = chain.run({"topic": "love"})
print(poem)
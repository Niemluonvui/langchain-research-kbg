# Anthropic
# pip install -U langchain-anthropic

from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model='claude-3-opus-20240229')

# Google
# pip install langchain-google-genai

# configure ID
# export GOOGLE_API_KEY=your-api-key

# gemini
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-pro")
llm.invoke("Sing a ballad of LangChain.")

# gemini-pro-vision
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-pro-vision")

message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "What's in this image?",
        },  # You can optionally provide text parts
        {"type": "image_url", "image_url": "https://picsum.photos/seed/picsum/200/300"},
    ]
)
llm.invoke([message])

# Huggingface model
# pip install langchain-huggingface transformers

from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# connecting model on huggingface
generator = pipeline("text-generation", model="gpt2")
llm = HuggingFacePipeline(pipeline=generator)

# Microsoft chat model

# pip install langchain-openai

import os
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://<your-endpoint.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "your AzureOpenAI key"

from langchain_openai import AzureChatOpenAI

AzureChatOpenAI(
    azure_deployment="35-turbo-dev",
    openai_api_version="2023-05-15",
)
# specify the version of the model using model_version constructor parameter

# OpenAI

# pip install -qU langchain-openai

import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4")
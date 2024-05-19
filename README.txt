This is Langchain Research repository where I put things to know about Langchain

main.py is sample of using model and adding prompt with keyword to generate the answer from the model

test2.pyv is sample as basic concept of using model to chat with systemMessage and HumanMessage

modeluse is sample of how to connect different AI platform

server.py is sample of using fastapi server for models

vectorstored.py is the example how store data/context into chromadb and llm gonna use it in answer

Some notices:

1. Make sure any AI you use, you must setting the global variables as Id or Key to be able to connect them with your project
Example:
import os
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://<your-endpoint.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "your AzureOpenAI key"

2. Hugging face got alot models that required GPU (CUDA cores) and more external libraries

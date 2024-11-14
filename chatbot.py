import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.output_parsers.string import StrOutputParser

load_dotenv()

LANGCHAIN_API_KEY = os.environ['LANGCHAIN_API_KEY']
LANGCHAIN_TRACING_V2 = os.environ['LANGCHAIN_TRACING_V2']

chatbot = ChatOllama(
  model='llama2'
)

chatbot = chatbot | StrOutputParser()

print(chatbot.invoke('hello'))
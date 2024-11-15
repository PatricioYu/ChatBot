import os, json, re, bs4, asyncio
from uuid import uuid4
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb

async def load_documents(urls: list[str]) -> list[str]:
  loader: WebBaseLoader = WebBaseLoader(web_paths=urls)
  docs: list[str] = []

  async for doc in loader.alazy_load():
    docs.append(doc)

  return docs

async def docs_to_clean_string(docs: list[str]) -> str:
  docString = ""

  RE_CLEANUP_NEWLINES: re.Pattern[str] = re.compile(r"(?<=[a-zA-Z])\n(?=[a-zA-Z])|\n{2,}")

  docString = "".join(doc.page_content for doc in docs)

  docString = RE_CLEANUP_NEWLINES.sub(lambda m: " " if m.group(0) == "\n" else "\n", docString)

  return docString

async def main():
  load_dotenv()

  LANGCHAIN_API_KEY = os.environ['LANGCHAIN_API_KEY']
  LANGCHAIN_TRACING_V2 = os.environ['LANGCHAIN_TRACING_V2']
  PATHS = json.loads(os.environ['PATHS'])

  chatbot = ChatOllama(
    model='llama2',
    temperature=0
  )

  docs = await load_documents(PATHS)
  docs = await docs_to_clean_string(docs)

  print(docs)

# embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# embeddings = embeddings_model.embed_documents(docs_text)

# vector_store = Chroma(
#     persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
# )

# uuids = [str(uuid4()) for _ in range(len(docs_text))]

# vector_store.add_documents(documents=docs_text, ids=uuids)

# print(vector_store.get())

# prompt = ChatPromptTemplate.from_messages(
#   [
#     ('system', 'you like to use emojis'),
#     ('human', '{input}')
#   ]
# )

# chain = prompt | chatbot | StrOutputParser()

# print(chain.invoke({'input': 'what is promptior'}))

# Run the event loop to start the process
if __name__ == "__main__":
    asyncio.run(main())  # This will run the main function and start the event loop
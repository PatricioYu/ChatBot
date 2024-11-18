import os, json, re, asyncio
from uuid import uuid4
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

async def load_documents(urls: list[str]) -> list[str]:
  loader: SeleniumURLLoader = SeleniumURLLoader(urls=urls)
  docs: list[Document] = []

  async for doc in loader.alazy_load():
    docs.append(doc)

  return docs

async def docs_to_clean_string(docs: list[Document]) -> str:
  docString: str = ""

  RE_CLEANUP_NEWLINES: re.Pattern[str] = re.compile(r"(?<=[a-zA-Z])\n(?=[a-zA-Z])|\n{2,}")

  docString = "".join(doc.page_content for doc in docs)

  docString = RE_CLEANUP_NEWLINES.sub(lambda m: " " if m.group(0) == "\n" else "\n", docString)

  return docString

def remove_redundant_text(text: str) -> str:
  """
  Removes exact duplicate sentences from the input text.
    :param text: The input text to deduplicate.
    :return: Cleaned text with duplicate sentences removed.
  """

  # Split the text into sentences (or paragraphs)
  sentences = text.split('\n')

  # Remove duplicates by converting the list of sentences to a set
  unique_sentences = list(set(sentences))

  # Join the unique sentences back into a string
  return "\n".join(unique_sentences)


async def split_str(string: str) -> list[str]:
  text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
  )

  chunks = text_splitter.split_text(string)

  return chunks

async def main():
  load_dotenv()

  LANGCHAIN_API_KEY: str = os.environ['LANGCHAIN_API_KEY']
  LANGCHAIN_TRACING_V2: str = os.environ['LANGCHAIN_TRACING_V2']
  PATHS: list[str] = json.loads(os.environ['PATHS'])

  chatbot = ChatOllama(
    model='llama2',
    temperature=0
  )

  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
  vector_store = Chroma(
      embedding_function=embeddings,
      persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
  )

  docs: list[str] = await split_str(remove_redundant_text(await docs_to_clean_string(await load_documents(PATHS))))
  metadata = [{"id": str(uuid4()), "source": "document1", "chunk": i} for i in range(len(docs))]
  print(f"Total documents in vector store: {len(docs)}")

  await vector_store.aadd_texts(docs, metadatas=metadata)

  results = vector_store.similarity_search_by_vector(
    embedding=embeddings.embed_query("Overcome"), k=2
  )
  for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")

# print(chain.invoke({'input': 'what is promptior'}))

# Run the event loop to start the process
if __name__ == "__main__":
    asyncio.run(main())  # This will run the main function and start the event loop
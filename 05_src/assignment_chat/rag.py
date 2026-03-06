import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain.tools import tool
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_community.vectorstores import Chroma
# from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings



load_dotenv('../.secrets')
file_path = './dataset/msaa-guide-to-programs-services-for-seniors-fall-en-2024-11-27.pdf'

# ----------- Load PDF Document -----------
loader = PyPDFLoader(file_path)
docs = loader.load()

#----------- Chunking the document -----------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap=200,
    length_function = len,
    add_start_index = True
)

chunks = text_splitter.split_documents(docs)


if not os.environ.get("API_GATEWAY_KEY"):
    raise ValueError("Missing API_GATEWAY_KEY environment variable")

persistent_client =  chromadb.PersistentClient(path="./vectordb/chroma_db")

persistent_collection = persistent_client.get_or_create_collection(name="senior_programs")
persistent_collection.add(
    documents=[chunk.page_content for chunk in chunks],
    metadatas=[chunk.metadata for chunk in chunks],
    ids=[str(i) for i in range(len(chunks))]
)

@tool
def retrieval_tool(query):
    """
    A retrieval tool that takes a user query as input and retrieves relevant information from the document chunks.
    The tool should return the retrieved information in a format that can be easily consumed by the LLM.
    """
    response = persistent_collection.query(
        query_texts=[query],
        n_results=2
    )
    return response
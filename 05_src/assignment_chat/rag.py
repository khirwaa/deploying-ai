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

# vector_store = Chroma(
#     collection_name="senior_programs",
#     embedding_function=OpenAIEmbeddingFunction(
#         api_key=os.getenv('API_GATEWAY_KEY'),
#         base_url='https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1'
#     ),
#     persist_directory="./vectordb/chroma_db",  # Where to save data locally, remove if not necessary
# )
# vector_store.add_documents(chunks)
persistent_client =  chromadb.PersistentClient(path="./vectordb/chroma_db")

persistent_collection = persistent_client.get_or_create_collection(name="senior_programs")
persistent_collection.add(
    documents=[chunk.page_content for chunk in chunks],
    metadatas=[chunk.metadata for chunk in chunks],
    ids=[str(i) for i in range(len(chunks))]
)
# return persistent_collection

# vector_store.add_documents(documents=chunks)
# return vector_store


# # ----------- Embed the document in a vector database -----------
# client = OpenAI(base_url='https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1',
#                 api_key='any value',
#                 default_headers={"x-api-key": os.getenv('API_GATEWAY_KEY')})

# def get_embedding(text, model="text-embedding-3-small"):
#     # text = text.replace("\n", " ")
#     return client.embeddings.create(input=text, model=model).data[0].embedding


# persistent_client = chromadb.PersistentClient(path="./RAG/chroma_db")
# # Create or get a collection; this data will be saved to disk
# persistent_collection = persistent_client.get_or_create_collection(name="senior_programs")

# persistent_collection.add(
#     documents=[chunk.page_content for chunk in chunks],
#     metadatas=[chunk.metadata for chunk in chunks],
#     ids=[str(i) for i in range(len(chunks))]
# )

# --------------------------
# --------------------------

@tool
def retrieval_tool(query):
    # retrieved_docs = vector_store.similarity_search(query, k=2)
    # serialized = "\n\n".join(
    #     (f"Source: {doc.metadata}\nContent: {doc.page_content}")
    #     for doc in retrieved_docs
    # )
    # return serialized, retrieved_docs
    """
    A retrieval tool that takes a user query as input and retrieves relevant information from the document chunks.
    The tool should return the retrieved information in a format that can be easily consumed by the LLM.
    """
    response = persistent_collection.query(
        query_texts=[query],
        n_results=2
    )
    return response
    #     )
    # for i, res in enumerate(response):
    #     print(f"Result {i+1}:")
    #     print(f"Content: {res.page_content}")
    #     print(f"Metadata: {res.metadata}\n")
    # return response
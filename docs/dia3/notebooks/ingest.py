from dotenv import load_dotenv
import argparse

from langchain.document_loaders.pdf import PyPDFDirectoryLoader 
from langchain.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_chroma import Chroma # This is a Chroma wrapper from Langchain
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()  # Load environment variables from a .env file if present

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # Local model

def ingest(pdf_path: str):

    loader = PyPDFLoader(pdf_path) # Tool to load and process a PDF file
    pdf_documents = loader.load() # Each document corresponds actually to a page
    print(len(pdf_documents), "loaded")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)

    texts = text_splitter.split_documents(pdf_documents)
    print(len(texts), "chunks")

    # We use a simple vector store for the chunks
    vectorstore_chroma = Chroma(
            collection_name="project_collection",
            embedding_function=embeddings,
            persist_directory="./project_chroma_db" # Optional: specify a directory to persist your data
        )
    vectorstore_chroma.add_documents(texts)



parser = argparse.ArgumentParser(description="...")
parser.add_argument('-d', '--pdf_path', type=str, required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    print("Ingesting PDF document...", args.pdf_path)
    ingest(args.pdf_path)
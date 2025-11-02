from fastmcp import FastMCP

from dotenv import load_dotenv
import os 

from langchain_chroma import Chroma # This is a Chroma wrapper from Langchain
from langchain_openai import ChatOpenAI # Import OpenAI LLM
from langchain import hub
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

import argparse

from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

TEMPLATE = """You are a helpful AI assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know, don't try to make up an answer

    Question: {question}
   
    Context: {context}

    Answer (answer in Spanish, only if question is in Spanish):
"""

load_dotenv()  # Load environment variables from .env file

os.environ["TOKENIZERS_PARALLELISM"] = "false"



mcp = FastMCP("docrag-server")

# initialize once
rag_prompt = hub.pull("rlm/rag-prompt", include_model=True)
# rag_prompt = ChatPromptTemplate.from_template(TEMPLATE)


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # Local model

llm_model = os.environ["OPENAI_MODEL"]
llm = ChatOpenAI(model=llm_model, temperature=0.1)

vectorstore_chroma = Chroma(
        collection_name="project_collection",
        embedding_function=embeddings,
        persist_directory="./project_chroma_db" # Optional: specify a directory to persist your data
    )

retriever = vectorstore_chroma.as_retriever(search_kwargs={"k": 7})

rag_chain = {"context": retriever,  "question": RunnablePassthrough()} | rag_prompt | llm | StrOutputParser()

@mcp.tool(description="Search architecture Surrogates scientific documentation and summarize relevant info")
def search_docs(query: str) -> str:
    # 
    result = rag_chain.invoke(query)
    #
    return result.strip()


if __name__ == "__main__":
    mcp.run(transport="http", port=8000)
    # mcp.run()

# parser = argparse.ArgumentParser(description="...")
# parser.add_argument('-q', '--question', type=str, required=True)

# if __name__ == "__main__":
#     args = parser.parse_args()
#     print("Running Python script ...") #, args)

#     if args.question:
#         print(search_docs(args.question))

#     print("--- Done ---")

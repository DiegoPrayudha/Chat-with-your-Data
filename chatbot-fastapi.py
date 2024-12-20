from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq  
from langchain_core.documents import Document
from typing import List
from dotenv import load_dotenv
import os

from fastapi import FastAPI
load_dotenv()

app = FastAPI()

# Groq API Key (optional, can be replaced with another model)
api_key = os.getenv('API_KEY')

# Model and embedding configuration
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
CHROMA_PATH = './chroma_db'

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

# Load and process the dataset
folder_path = 'data/used_car_dataset.csv'
loader = CSVLoader(file_path = folder_path, encoding="utf-8")
documents = loader.load_and_split()


vector_store = Chroma(
    collection_name="car_dataset",
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,  # Where to save data locally, remove if not necessary
)

def add_documents_in_batches(vector_store, documents: List[Document], batch_size: int = 5000):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        vector_store.add_documents(documents=batch)

# Ganti `vector_store.add_documents(documents=documents)` dengan:
add_documents_in_batches(vector_store, documents, batch_size=5461)

# vector_store.add_documents(documents=documents)

# Create a prompt template
prompt = ChatPromptTemplate([
    ("system", """
        You are an Helpful Assistant. The user will provide you with a CSV file and a question. You will answer the question based on the data in the CSV file.
        """),
    ("human","""
        This is context from csv file:
        {context}
        This is the question:
        {question}
        """ ),
])

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vector_store.as_retriever(),
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt}
)

@app.get("/")
async def read_root():
    return {"status": "active", "message": "Welcome to the Used Car QA System"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/query")
async def query_langchain(query: str):
    answer = qa_chain.run(query)
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
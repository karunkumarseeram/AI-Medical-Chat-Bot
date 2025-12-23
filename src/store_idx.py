from dotenv import load_dotenv
import os
from src.helper import load_pdf_files,filer_to_minimal_docs,text_split,download_hugging_face_embeddings
from pinecone import Pinecone
from pinecone import Pinecone,ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from pathlib import Path



# Load environment variables

load_dotenv()


PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("Missing API keys. Please set PINECONE_API_KEY and OPENAI_API_KEY in your .env file.")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Resolve data folder path safely


BASE_DIR = Path(__file__).resolve().parent   # points to src/ if this file is inside src/
DATA_DIR = BASE_DIR.parent / "data"          # go up one level, then into data/

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Data folder not found at {DATA_DIR}. Please create it and add your PDFs.")


# Load and preprocess PDFs

extracted_data = load_pdf_files(data=str(DATA_DIR))
filter_data = filer_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

#  Load embeddings
embeddings = download_hugging_face_embeddings()


# Setup Pinecone

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

index_name = "medical-chaatbot" #change it if you wnat as per your requirement to be

#create a idex if not be present in vector db

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws",region="us-east-1")
    )
index = pc.Index(index_name)


# Upload documents to Pinecone

doc_search = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

print("Documents successfully uploaded to Pinecone index:", index_name)



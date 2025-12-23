from langchain.document_loaders import DirectoryLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document


# Extracted data from the PDF files
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


# Filter to Minimal doc texts from the source data
def filer_to_minimal_docs(docs:List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_contnent.
"""
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source":src}
            )
        )
    return minimal_docs


#Split the docs(Data) into smaller text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks
    
    
#Download the Embeddings from HuggingFace
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

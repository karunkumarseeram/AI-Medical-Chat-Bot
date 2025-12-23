from flask import Flask, render_template,jsonify,request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
import os


app = Flask(__name__)

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY



embeddings = download_hugging_face_embeddings()


index_name = "medical-chatbot" #change it if you wnat as per your requirement to be

doc_search = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


# decelop the chain code to the get the data from vector pinecone db

retriever = doc_search.as_retriever(search_type = "similarity",search_kwargs={"k":2})

chatModel = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{input}")
    ]
)

#create a question & answer chain
question_answer_chain = create_stuff_documents_chain(chatModel,prompt)
#create rag cahin from the retreiver from question
rag_chain = create_retrieval_chain(retriever,question_answer_chain)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get",methods=["GET","POST"])
def chat():
    msg=request.form["msg"]
    input = msg
    print(input)
    res = rag_chain.invoke({"input":msg})
    print("Response : ",res["answer"])
    return str(res["answer"])


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)
from flask import Flask, render_template, request, jsonify
import os
import bs4
import pandas as pd
from PyPDF2 import PdfReader
from langchain import hub
from langchain_community.vectorstores import FAISS  
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_groq import ChatGroq

app = Flask(__name__)

os.environ["GROQ_API_KEY"] = "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"

pdf_path = "D:/GENAI PRACTICE/Embeddings/apiic_scraped.pdf"
pdf_reader = PdfReader(pdf_path)
pdf_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

csv_path = "D:/GENAI PRACTICE/Embeddings/apiic_links.csv"
links_df = pd.read_csv(csv_path)
links_text = "\n".join(links_df["Link"].astype(str))

data = pdf_text + "\n" + links_text
doc = Document(page_content=data)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents([doc], embedding=embeddings)
vector_store.save_local("faiss_index")


loader = WebBaseLoader(
    web_paths=("https://apiic.in/",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

if all_splits:
    vector_store.add_documents(documents=all_splits)

prompt = hub.pull("rlm/rag-prompt")
llm = ChatGroq(model_name="mixtral-8x7b-32768", temperature=0)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"], k=10) 
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    docs_content = docs_content[:10000]
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get("question")
    response = graph.invoke({"question": question})
    return jsonify({"answer": response["answer"]})

if __name__ == '__main__':
    app.run(debug=True)




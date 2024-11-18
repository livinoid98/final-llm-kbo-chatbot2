from fastapi import FastAPI
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch
from langchain import hub
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOpenAI
import openai
import os
from langfuse.callback import CallbackHandler
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

@app.post("/")
async def root(question: Question):
    loader = WebBaseLoader("https://sports.news.naver.com/kbaseball/news/index?isphoto=N")
    docs = loader.load()

    print(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50
    )

    embeddings = HuggingFaceEmbeddings(model_name="dragonkue/bge-m3-ko")

    es_url = "http://127.0.0.1:9200"
    index_name = "news_articles"
    ES_HOST = 'http://127.0.0.1:9200'

    es = Elasticsearch(
        hosts=[ES_HOST],
        verify_certs=False
    )

    database = ElasticsearchStore(
        es_url=es_url,
        index_name=index_name,
        embedding=embeddings,
        es_connection=es
    )

    for i in range(0, len(docs)):
        database.add_documents(text_splitter.split_documents(docs[i:i+1]))

    retriever = database.as_retriever()

    prompt = hub.pull("rlm/rag-prompt")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    langfuse_handler = CallbackHandler(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host="http://127.0.0.1:8000"
    )

    query = question
    response = rag_chain.invoke(query.question, config={"callbacks": [langfuse_handler]})
    
    return {"answer" : response}
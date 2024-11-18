from fastapi import FastAPI
from pydantic import BaseModel
import os
import openai
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import torch
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
load_dotenv()

origins = [
    "http://localhost:3000", 
    "http://127.0.0.1:3000",
    "http://localhost:8080"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],     
    allow_headers=["*"],     
)

class Question(BaseModel):
    input: str

@app.post("/question")
def root(question: Question):
    print(question)
    openai.api_key = os.getenv("OPENAI_API_KEY")

    url = 'https://sports.news.naver.com/kbaseball/news/index?isphoto=N'
    loader = WebBaseLoader(url)
    docs = loader.load()
    print(docs)
    baseballNews = docs[0].page_content

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = list(text_splitter.split_documents(docs))

    chat_prompt = ChatPromptTemplate.from_messages({
        ("system", "이 시스템은 야구와 관련한 질문에 답변할 수 있습니다."),
        ("user", question.input)
    })
    messages = chat_prompt.format_messages(user_input = "서스펜디드 게임")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    chain = chat_prompt | llm | StrOutputParser()
    # answer = chain.invoke({
    #     "user_input": question.input
    # })
    answer = "gpt가 생성한 응답"

    pc = Pinecone(
        api_key = os.getenv("PINECONE_API_KEY")
    )
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([split.page_content for split in splits])

    print('embeddings', embeddings)

    index_name = 'kbo-news'
    pc.delete_index(name=index_name)
    pc.create_index(
        name=index_name, 
        dimension=embeddings.shape[1],
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    index = pc.Index(index_name)
    
    ids = [f"id-{i}" for i in range(len(splits))]
    index.upsert(vectors=list(zip(ids, embeddings.tolist())), namespace="kbo_news")

    query_embedding = model.encode("사상 첫 ‘서스펜디드’…내일 재개")
    if isinstance(query_embedding, torch.Tensor):
        query_embedding = query_embedding.detach().numpy()

    pinecone_result = index.query(vector=query_embedding.tolist(), top_k=5, namespace="kbo_news")
    if pinecone_result['matches']:
        best_match = max(pinecone_result['matches'], key=lambda x: x['score'])
        best_match_id = best_match['id']
        best_match_score = best_match['score']
        best_match_content = splits[int(best_match_id.split('-')[1])].page_content  # id에서 인덱스 추출
    else:
        print("No matches found.")

    answer_embedding = model.encode(answer)

    pinecone_result = index.query(vector=answer_embedding.tolist(), top_k=10, namespace="kbo_news")
    pinecone_response = pinecone_result['matches']

    final_response = {
        "llm_answer": answer,
        "similar_docs": [match['metadata']['content'] for match in pinecone_response]
    }
    
    return final_response
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from dotenv import load_dotenv
from langchain_community.chat_models import QianfanChatEndpoint

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 加载模型和向量数据库
embeddings = ModelScopeEmbeddings(model_id='iic/nlp_corom_sentence-embedding_chinese-base')
vector_db = FAISS.load_local('LLM.faiss', embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# 配置聊天模型和Prompt模板
llmchat = QianfanChatEndpoint(model="ERNIE-4.0-8K", temperature=0.5)
system_prompt = SystemMessagePromptTemplate.from_template('You are a helpful assistant.')
user_prompt = HumanMessagePromptTemplate.from_template('''
Answer the question based only on the following context:

{context}

Question: {query}
''')
full_chat_prompt = ChatPromptTemplate.from_messages([system_prompt, MessagesPlaceholder(variable_name="chat_history"), user_prompt])

chat_chain = {
    "context": itemgetter("query") | retriever,
    "query": itemgetter("query"),
    "chat_history": itemgetter("chat_history"),
} | full_chat_prompt | llmchat

class ChatRequest(BaseModel):
    query: str
    chat_history: List[Dict[str, Any]] = []

class ChatResponse(BaseModel):
    response: str
    chat_history: List[Dict[str, str]]

@app.get("/", response_class=HTMLResponse)
async def get_chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request, "chat_history": []})

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    chat_history = [HumanMessage(content=msg['content']) if msg['role'] == 'user' else AIMessage(content=msg['content'])
                    for msg in request.chat_history]
    response = chat_chain.invoke({'query': request.query, 'chat_history': chat_history})
    chat_history.extend([HumanMessage(content=request.query), response])
    chat_history = chat_history[-20:]  # 保留最近20条对话
    return ChatResponse(
        response=response.content,
        chat_history=[{'role': 'user', 'content': msg.content} if isinstance(msg, HumanMessage) else {'role': 'assistant', 'content': msg.content}
                      for msg in chat_history]
    )

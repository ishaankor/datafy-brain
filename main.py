import os
import sys
import io
import uvicorn
import logging
from contextlib import redirect_stdout
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("datafy-backend")

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    datasetContext: Optional[str] = ""
    selectionCSV: Optional[str] = ""
    selectionLabel: Optional[str] = ""

repl_locals = {}

@tool
def python_repl_tool(command: str) -> str:
    """A Python shell. Use this to execute python commands."""
    logger.info(f"🛠️  AGENT IS EXECUTING: {command}")
    f = io.StringIO()
    with redirect_stdout(f):
        try:
            exec(command, globals(), repl_locals)
        except Exception as e:
            logger.error(f"❌ EXECUTION ERROR: {e}")
            return f"{f.getvalue()}\nError: {e}".strip()
    
    result = f.getvalue().strip()
    logger.info(f"✅ EXECUTION RESULT: {result}")
    return result

tools = [python_repl_tool]

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

system_prompt = """You are an elite Data Scientist and Statistician. Use Python REPL to find statistical truth."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    logger.info(f"📨 RECEIVED CHAT REQUEST: {req.selectionLabel}")
    try:
        active_csv = req.selectionCSV if req.selectionCSV else req.datasetContext
        if active_csv:
            logger.info("💾 SAVING CURRENT_DATA.CSV")
            with open("current_data.csv", "w") as f:
                f.write(active_csv)

        langchain_msgs = []
        for msg in req.messages:
            if msg.role == "user":
                langchain_msgs.append(("human", msg.content))
            elif msg.role == "assistant":
                langchain_msgs.append(("ai", msg.content))
        
        context_note = f"\n\n[SYSTEM: Analyze the data in current_data.csv.]"
        if langchain_msgs and langchain_msgs[-1][0] == "human":
            langchain_msgs[-1] = ("human", langchain_msgs[-1][1] + context_note)

        logger.info("🧠 AGENT STARTING THOUGHT CHAIN...")
        result = agent_executor.invoke({"messages": langchain_msgs})
        logger.info("🎯 AGENT FINISHED. RETURNING OUTPUT.")
        
        return {"response": result["output"]}
    
    except Exception as e:
        logger.error(f"🚨 CRITICAL ENDPOINT ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))
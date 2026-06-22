import os
import io
import uvicorn
import logging
from contextlib import redirect_stdout
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import json

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("datafy-backend")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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
    """Execute Python code. Output must be print statements."""
    f = io.StringIO()
    with redirect_stdout(f):
        try:
            exec(command, globals(), repl_locals)
        except Exception as e:
            return f"Error: {e}"
    return f.getvalue().strip()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an elite Data Scientist. Use Python REPL for math. DO NOT use plotting libraries."),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, [python_repl_tool], prompt)
executor = AgentExecutor(agent=agent, tools=[python_repl_tool], verbose=True)

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    if req.selectionCSV or req.datasetContext:
        with open("current_data.csv", "w") as f:
            f.write(req.selectionCSV or req.datasetContext or "")

    async def generate():
        langchain_msgs = [(m.role if m.role != "assistant" else "ai", m.content) for m in req.messages]
        
        result = await executor.ainvoke({"messages": langchain_msgs})
        output = result["output"]
        
        yield f"0:{json.dumps(output)}\n"

    return StreamingResponse(generate(), media_type="text/plain")
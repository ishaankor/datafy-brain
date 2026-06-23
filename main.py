import os
import sys
import io
import pandas as pd
import json
from contextlib import redirect_stdout
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool

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
    """A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`."""
    f = io.StringIO()
    with redirect_stdout(f):
        try:
            exec(command, globals(), repl_locals)
        except Exception as e:
            return f"{f.getvalue()}\nError: {e}".strip()
    return f.getvalue().strip()

tools = [python_repl_tool]

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

system_prompt = """You are an elite Data Scientist and Statistician. You have access to a fully functional Python REPL.
Instead of guessing math, you MUST write and execute Python code using `pandas` and `numpy` to find the exact statistical truth.

The user's current data selection has been saved locally as `current_data.csv`.
ALWAYS load the data in your first thought using:
```python
import pandas as pd
df = pd.read_csv('current_data.csv')
```

## RULES
1. ALWAYS target mathematical concepts and explain the math deeply based on the exact results of your Python execution.
2. DO NOT try to calculate variance, standard deviation, or complex math in your head. Write a script, run it, and read the output.
3. DO NOT generate matplotlib or seaborn plots. The user's frontend has a custom rendering engine.
4. When a chart would help visualize the data, output a fenced code block tagged as 'chart' containing JSON of this exact shape:
```chart
{{ "type": "line"|"bar"|"pie"|"scatter"|"area", "title": "...", "caption": "...", "colors": ["#F5D061", "#E8912E", "#F8B150"], "x": "<x-field>", "y": "<y-field>" or ["y1","y2"], "z": "<z-field> (optional)", "data": [{{"name": "Row 5", "<x-field>": ...}}] }}
```
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        active_csv = req.selectionCSV if req.selectionCSV else req.datasetContext
        if active_csv:
            with open("current_data.csv", "w") as f:
                f.write(active_csv)
        
        async def generate():
            langchain_msgs = [(m.role if m.role != "assistant" else "ai", m.content) for m in req.messages]
            
            context_note = f"\n\n[SYSTEM: The active data context is '{req.selectionLabel}'. The file 'current_data.csv' has been updated with this data. Analyze this data to answer what the USER is asking for.]"
            if langchain_msgs and langchain_msgs[-1][0] == "human":
                langchain_msgs[-1] = ("human", langchain_msgs[-1][1] + context_note)
            
            result = await agent_executor.ainvoke({"messages": langchain_msgs})
            output = result["output"]
            
            yield f"0:{json.dumps(output)}\n"

        return StreamingResponse(generate(), media_type="text/plain")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
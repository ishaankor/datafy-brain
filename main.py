import os
import sys
import io
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

system_prompt = """You are an autonomous, elite Data Scientist and Statistician. You have access to a Python REPL.
You do not guess; you compute. ALWAYS target mathematical concepts and explain the math deeply based on exact results.

The user's current data selection has been saved locally as `current_data.csv`.
ALWAYS load the data in your first thought using:
```python
import pandas as pd
df = pd.read_csv('current_data.csv')
```

## YOUR AUTONOMY & APPROACH
You have complete autonomy over how you analyze the data. You are not bound to a specific workflow. 
1. Read the user's specific query carefully.
2. Write Python code to explore, transform, aggregate, or model the dataset as necessary to find the mathematically rigorous answer.
3. If the user's question is open-ended (e.g., "Analyze this"), write code to programmatically discover the most mathematically interesting patterns, correlations, or distributions.
4. Formulate your final response by explaining the statistical truth you discovered.

## JSON CHART PROTOCOL (STRICT)
Do NOT generate matplotlib or seaborn plots. The user's frontend uses a custom rendering engine.
If your analysis warrants a visualization to explain the mathematical concepts, output a fenced code block tagged as 'chart' containing JSON of this exact shape:
```chart
{{ "type": "line"|"bar"|"pie"|"scatter"|"area", "title": "...", "caption": "...", "colors": ["#F5D061", "#E8912E", "#F8B150"], "x": "<primary-x-field>", "y": "<primary-y-field>" or ["y1","y2"], "z": "<third-dimension-field> (optional)", "data": [{{"name": "Point 1", "<x-field>": ..., "<y-field>": ...}}] }}
```
Choose the chart type that mathematically best represents the data relationships (e.g., scatter for continuous correlation, z-axis for a third dimension, bar for categorical comparison).
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

        langchain_msgs = []
        for msg in req.messages:
            if msg.role == "user":
                langchain_msgs.append(("human", msg.content))
            elif msg.role == "assistant":
                langchain_msgs.append(("ai", msg.content))
        
        context_note = f"\n\n[SYSTEM: The active data context is '{req.selectionLabel}'. The file 'current_data.csv' has been updated with this data. Analyze this data to what the USER is asking for.]"
        if langchain_msgs and langchain_msgs[-1][0] == "human":
            langchain_msgs[-1] = ("human", langchain_msgs[-1][1] + context_note)

        result = agent_executor.invoke({"messages": langchain_msgs})
        
        return {"response": result["output"]}
    
    except Exception as e:
        print(f"CRASH: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))
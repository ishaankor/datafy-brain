import os
import sys
import io
import re
import json
import base64
import operator
import logging
from contextlib import redirect_stdout
from typing import List, Optional, TypedDict, Annotated, Sequence

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent

# ---------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("DataCopilot")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

primary_fast_llm = ChatOpenRouter(
    model="nemotron-3-super-120b-a12b:free", 
    temperature=0.1
)

backup_smart_llm = ChatOpenRouter(
    model="nemotron-3-nano-30b-a3b:free", 
    temperature=0.1
)

llm = primary_fast_llm.with_fallbacks([backup_smart_llm])
writer_llm = ChatOpenRouter(
    model="gpt-oss-20b:free", 
    temperature=0.1
).with_fallbacks([backup_smart_llm])

# ---------------------------------------------------------
# 2. STRUCTURED DATA MODELS & STATE
# ---------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_query: str
    dataset_profile: str
    statistical_results: str
    image_artifacts: List[str]

# ---------------------------------------------------------
# 3. TOOLS
# ---------------------------------------------------------
repl_locals = {}

@tool
def python_repl_tool(command: str) -> str:
    """Executes arbitrary Python code. Use this for pandas manipulations, statistical tests, and ML models.
    Always print() the final values you want to observe."""
    logger.info(f"Executing Python Code:\n{command}")
    f = io.StringIO()
    with redirect_stdout(f):
        try:
            exec(command, repl_locals, repl_locals)
        except Exception as e:
            logger.error(f"Python Error: {e}")
            return f"{f.getvalue()}\nError: {e}".strip()
    
    output = f.getvalue().strip()
    logger.info(f"Python Output:\n{output[:500]}...")
    return output

tools = [python_repl_tool]

# ---------------------------------------------------------
# 4. SPECIALIZED REACT AGENTS
# ---------------------------------------------------------
stat_agent = create_agent(llm, tools)

# ---------------------------------------------------------
# 5. WORKFLOW NODES
# ---------------------------------------------------------

def profiler_node(state: AgentState):
    logger.info("--- NODE: PROFILER ---")
    try:
        df = pd.read_csv("current_data.csv")
        profile = {
            "rows": len(df),
            "columns": len(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isna().sum().to_dict(),
            "numeric_columns": df.select_dtypes("number").columns.tolist(),
            "categorical_columns": df.select_dtypes(exclude="number").columns.tolist(),
            "basic_summary": df.describe().to_dict()
        }
        profile_str = json.dumps(profile, indent=2)
        logger.info(f"Dataset profiled successfully: {len(df)} rows, {len(df.columns)} columns.")
    except Exception as e:
        profile_str = f"Error profiling dataset: {str(e)}"
        logger.error(profile_str)
    
    return {"dataset_profile": profile_str}

def unified_executor_node(state: AgentState):
    logger.info("--- NODE: UNIFIED STRATEGIC EXECUTOR ---")
    
    sys_prompt = f"""You are an autonomous Principal Data Scientist.
    DATASET PROFILE: {state['dataset_profile']}
    USER QUERY: {state['user_query']}
    
    You have complete analytical freedom to solve the user's query. The data is available locally at `current_data.csv`. 
    
    Depending on the mathematical concepts and logical requirements of the query, you must autonomously decide whether to:
    - Perform exploratory data analysis (EDA) and print summary statistics.
    - Run inferential statistics (e.g., t-tests, correlations).
    - Train and evaluate machine learning models (e.g., exploring ensemble methods like joining decision trees with random_forests).
    - Create visualizations.
    
    CRITICAL EXECUTION RULES:
    1. Respond directly to the intent. Evaluate the true mathematical requirements. If the query is just "Compare these", run a statistical comparison or print a pandas summary. DO NOT blindly generate a plot unless it explicitly serves the analytical goal.
    2. Treat the CSV file as read-only. Do not overwrite `current_data.csv`, as those dataset inputs are utilized by other commands.
    3. If your analysis DOES require a visualization, you MUST save it to disk strictly as `chart.png`. DO NOT output base64 strings to the console.
    4. Print all numerical conclusions and statistical summaries clearly so the Final Delivery Agent can read them.
    5. Always generate questions that target mathematical concepts to guide the user's understanding of the data if further clarification is needed.
    
    Write and execute the appropriate Python code now."""

    res = stat_agent.invoke({"messages": [SystemMessage(content=sys_prompt), HumanMessage(content="Analyze the dataset and run the code.")]})
    
    current_images = state.get("image_artifacts", [])
    
    if os.path.exists("chart.png"):
        with open("chart.png", "rb") as img_file:
            current_images.append(base64.b64encode(img_file.read()).decode("utf-8"))
        os.remove("chart.png")
        
    return {
        "messages": [],
        "statistical_results": res['messages'][-1].content,
        "image_artifacts": current_images
    }

def writer_node(state: AgentState):
    logger.info("--- NODE: WRITER ---")
    sys_prompt = f"""You are the Final Delivery Agent.
    USER QUERY: {state['user_query']}
    EXECUTION RESULTS: {state.get('statistical_results', '')}
    
    CRITICAL INSTRUCTIONS:
    1. Do not attempt to write the markdown image tag yourself. It is handled by the system.
    2. DO NOT apologize for missing base64 strings or missing data.
    3. If the execution results indicate an image was generated and there is no other mathematical data, simply say: "Here is the visualization you requested:" and stop.
    
    Synthesize the mathematical findings smoothly and clearly."""
    
    response = writer_llm.invoke([SystemMessage(content=sys_prompt)])
    logger.info("Final response generated successfully.")
    return {"messages": [AIMessage(content=response.content)]}

# ---------------------------------------------------------
# 6. GRAPH COMPILATION (Flattened)
# ---------------------------------------------------------
workflow = StateGraph(AgentState)

workflow.add_node("profiler", profiler_node)
workflow.add_node("executor", unified_executor_node)
workflow.add_node("writer", writer_node)

workflow.add_edge(START, "profiler")
workflow.add_edge("profiler", "executor")
workflow.add_edge("executor", "writer")
workflow.add_edge("writer", END)

app_graph = workflow.compile()

# ---------------------------------------------------------
# 7. FASTAPI ENDPOINT
# ---------------------------------------------------------
class ChatMessageReq(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessageReq]
    datasetContext: Optional[str] = ""
    selectionCSV: Optional[str] = ""
    selectionLabel: Optional[str] = ""

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    logger.info("=== NEW CHAT REQUEST RECEIVED ===")
    try:
        active_csv = req.selectionCSV if req.selectionCSV else req.datasetContext
        if active_csv:
            with open("current_data.csv", "w") as f:
                f.write(active_csv)
            logger.info("Updated 'current_data.csv' with new active context.")

        user_query = next((msg.content for msg in reversed(req.messages) if msg.role == 'user'), "")
        logger.info(f"User Query: {user_query}")

        initial_state = {
            "messages": [],
            "user_query": f"Data Context: {req.selectionLabel}\nQuery: {user_query}",
            "dataset_profile": "",
            "statistical_results": "",
            "image_artifacts": []
        }

        logger.info("Invoking LangGraph Workflow...")
        config = {"recursion_limit": 10}
        final_state = app_graph.invoke(initial_state, config)
        
        if "messages" in final_state and len(final_state["messages"]) > 0:
            final_msg = final_state["messages"][-1]
            response_text = final_msg.content if hasattr(final_msg, 'content') else str(final_msg)
        else:
            response_text = "Here are the results of the analysis:"
            
        images = final_state.get("image_artifacts", [])
        if images:
            logger.info(f"Injecting {len(images)} base64 visual artifacts into response text.")
            response_text += "\n\n### Visualizations\n"
            for img in images:
                clean_img = img.replace("IMAGE_BASE64:", "").strip()
                response_text += f"\n![Generated Chart](data:image/png;base64,{clean_img})\n"
        
        logger.info("=== WORKFLOW COMPLETE ===")
        return {"response": response_text}
    
    except Exception as e:
        logger.error(f"CRASH in chat_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
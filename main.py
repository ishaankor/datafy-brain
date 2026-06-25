import os
import sys
import io
import json
from contextlib import redirect_stdout
from typing import List, Optional, TypedDict, Annotated, Sequence
import operator

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# ---------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------
load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatOpenRouter(
    model="nvidia/llama-nemotron-rerank-vl-1b-v2:free", 
    temperature=0.1
)

# ---------------------------------------------------------
# 2. STATE DEFINITION
# ---------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_query: str
    dataset_profile: str
    analysis_plan: str
    execution_results: str
    critic_feedback: str
    revision_count: int

# ---------------------------------------------------------
# 3. SPECIALIZED TOOLS & REPL
# ---------------------------------------------------------
repl_locals = {}

@tool
def python_repl_tool(command: str) -> str:
    """Executes arbitrary Python code. Use this for pandas manipulations, transformations, or calculations."""
    f = io.StringIO()
    with redirect_stdout(f):
        try:
            exec(command, globals(), repl_locals)
        except Exception as e:
            return f"{f.getvalue()}\nError: {e}".strip()
    return f.getvalue().strip()

@tool
def correlation_analysis(df_name: str, method: str = 'pearson') -> str:
    """Computes the correlation matrix for a given dataframe. 
    Args: df_name (name of the variable holding the dataframe), method ('pearson', 'spearman', or 'kendall')."""
    code = f"print(pd.DataFrame({df_name}).corr(numeric_only=True, method='{method}').to_string())"
    return python_repl_tool.invoke({"command": code})

tools = [python_repl_tool, correlation_analysis]
llm_with_tools = llm.bind_tools(tools)

# ---------------------------------------------------------
# 4. AGENT NODES
# ---------------------------------------------------------

def profiler_node(state: AgentState):
    """Automatically profiles the dataset before any LLM reasoning."""
    try:
        df = pd.read_csv("current_data.csv")
        profile = {
            "rows": len(df),
            "columns": len(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isna().sum().to_dict(),
            "numeric_columns": df.select_dtypes("number").columns.tolist(),
            "categorical_columns": df.select_dtypes(exclude="number").columns.tolist(),
        }
        profile_str = json.dumps(profile, indent=2)
    except FileNotFoundError:
        profile_str = "No dataset loaded or found at 'current_data.csv'."
    except Exception as e:
        profile_str = f"Error profiling dataset: {str(e)}"
    
    return {"dataset_profile": profile_str, "revision_count": 0}

def planner_node(state: AgentState):
    """Creates a mathematical analysis plan based on the profile and user query."""
    sys_prompt = f"""You are an elite Data Science Planner. 
    DATASET PROFILE:
    {state['dataset_profile']}
    
    USER QUESTION:
    {state['user_query']}
    
    Before writing any code, generate a step-by-step statistical and mathematical analysis plan.
    1. Identify variable types.
    2. Select the correct statistical framework (e.g., Regression, ANOVA, Time Series).
    3. Specify exactly what mathematical formulas or python tools need to be run.
    DO NOT output Python code. Output the PLAN only."""
    
    response = llm.invoke([SystemMessage(content=sys_prompt)])
    return {"analysis_plan": response.content}

def executor_node(state: AgentState):
    """Executes the plan using Python tools."""
    sys_prompt = f"""You are the Data Science Executor. 
    DATASET PROFILE: {state['dataset_profile']}
    ANALYSIS PLAN: {state['analysis_plan']}
    
    The dataset is saved locally as `current_data.csv`. 
    ALWAYS load it first: `import pandas as pd; df = pd.read_csv('current_data.csv')`.
    
    Execute the plan using your tools. If a tool returns an error, write new code to fix it.
    Output ONLY the final mathematical and statistical results computed by your tools."""
    
    messages = [SystemMessage(content=sys_prompt), HumanMessage(content="Execute the plan now.")]
    response = llm_with_tools.invoke(messages)
    
    execution_output = ""
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call['name'] == 'python_repl_tool':
                res = python_repl_tool.invoke(tool_call['args'])
                execution_output += f"Tool Output:\n{res}\n"
            elif tool_call['name'] == 'correlation_analysis':
                res = correlation_analysis.invoke(tool_call['args'])
                execution_output += f"Correlation Output:\n{res}\n"
    else:
        execution_output = response.content
        
    return {"execution_results": execution_output}

def critic_node(state: AgentState):
    """Verifies the statistical rigor and checks for hallucinations."""
    sys_prompt = f"""You are the Statistical Critic.
    USER QUERY: {state['user_query']}
    PLAN: {state['analysis_plan']}
    RESULTS: {state['execution_results']}
    
    Verify the results. 
    1. Were sample sizes mentioned?
    2. Are the conclusions statistically sound based on the results provided?
    3. Are there any obvious errors or hallucinated numbers?
    
    If the results are sound, output "APPROVED: " followed by a brief confirmation.
    If the results are flawed, output "REJECTED: " followed by exactly what the Executor must fix."""
    
    response = llm.invoke([SystemMessage(content=sys_prompt)])
    return {"critic_feedback": response.content, "revision_count": state["revision_count"] + 1}

def writer_node(state: AgentState):
    """Drafts the final response for the user, formatting charts if necessary."""
    sys_prompt = f"""You are the Final Writer.
    USER QUERY: {state['user_query']}
    VALIDATED RESULTS: {state['execution_results']}
    CRITIC NOTES: {state['critic_feedback']}
    
    Synthesize the findings into a clear, mathematically rigorous final response. 
    If a chart is needed, use the strict JSON chart protocol:
    ```chart
    {{ "type": "line"|"bar"|"pie"|"scatter"|"area", ... }}
    ```
    """
    response = llm.invoke([SystemMessage(content=sys_prompt)])
    return {"messages": [AIMessage(content=response.content)]}

# ---------------------------------------------------------
# 5. GRAPH ROUTING & COMPILATION
# ---------------------------------------------------------
def route_critic(state: AgentState):
    """Decides whether to loop back to the executor or proceed to the writer."""
    if "REJECTED" in state['critic_feedback'] and state['revision_count'] < 3:
        return "executor"
    return "writer"

workflow = StateGraph(AgentState)

workflow.add_node("profiler", profiler_node)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("critic", critic_node)
workflow.add_node("writer", writer_node)

workflow.add_edge(START, "profiler")
workflow.add_edge("profiler", "planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "critic")
workflow.add_conditional_edges("critic", route_critic, {"executor": "executor", "writer": "writer"})
workflow.add_edge("writer", END)

app_graph = workflow.compile()

# ---------------------------------------------------------
# 6. FASTAPI ENDPOINTS
# ---------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    datasetContext: Optional[str] = ""
    selectionCSV: Optional[str] = ""
    selectionLabel: Optional[str] = ""

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        active_csv = req.selectionCSV if req.selectionCSV else req.datasetContext
        if active_csv:
            with open("current_data.csv", "w") as f:
                f.write(active_csv)

        user_query = next((msg.content for msg in reversed(req.messages) if msg.role == 'user'), "")

        initial_state = {
            "messages": [],
            "user_query": f"Data Context: {req.selectionLabel}\nQuery: {user_query}",
            "dataset_profile": "",
            "analysis_plan": "",
            "execution_results": "",
            "critic_feedback": "",
            "revision_count": 0
        }

        final_state = app_graph.invoke(initial_state)
        
        return {"response": final_state["messages"][-1].content}
    
    except Exception as e:
        print(f"CRASH: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))
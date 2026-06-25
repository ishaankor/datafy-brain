import os
import sys
import io
import json
import operator
from contextlib import redirect_stdout
from typing import List, Optional, TypedDict, Annotated, Sequence, Any
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent

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
    model="nvidia/nemotron-3-ultra-550b-a55b:free", 
    temperature=0.1
)

# ---------------------------------------------------------
# 2. STRUCTURED DATA MODELS & STATE
# ---------------------------------------------------------
class AnalysisPlan(BaseModel):
    goal: str = Field(description="The primary mathematical or analytical goal.")
    variables: List[str] = Field(description="Key columns to investigate.")
    statistical_tests: List[str] = Field(description="Specific statistical tests to run (e.g., ANOVA, t-test, pearson correlation).")
    ml_tasks: List[str] = Field(description="Machine learning tasks (e.g., PCA, K-Means clustering, Feature Importance).")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_query: str
    dataset_profile: str
    analysis_plan: Optional[AnalysisPlan]
    statistical_results: str
    ml_results: str
    critic_feedback: str
    revision_count: int

# ---------------------------------------------------------
# 3. TOOLS
# ---------------------------------------------------------
repl_locals = {}

@tool
def python_repl_tool(command: str) -> str:
    """Executes arbitrary Python code. Use this for pandas manipulations, statistical tests, and ML models.
    Always print() the final values you want to observe."""
    f = io.StringIO()
    with redirect_stdout(f):
        try:
            exec(command, globals(), repl_locals)
        except Exception as e:
            return f"{f.getvalue()}\nError: {e}".strip()
    return f.getvalue().strip()

tools = [python_repl_tool]

# ---------------------------------------------------------
# 4. SPECIALIZED REACT AGENTS (Sub-Graphs)
# ---------------------------------------------------------

stat_agent = create_agent(llm, tools)
ml_agent = create_agent(llm, tools)

# ---------------------------------------------------------
# 5. WORKFLOW NODES
# ---------------------------------------------------------

def profiler_node(state: AgentState):
    """Deterministic EDA profiling. Generates the context for the LLM."""
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
    except Exception as e:
        profile_str = f"Error profiling dataset: {str(e)}"
    
    return {"dataset_profile": profile_str, "revision_count": 0}

def planner_node(state: AgentState):
    """Outputs a structured JSON plan for the downstream execution agents."""
    sys_prompt = f"""You are the Master Data Science Planner.
    DATASET PROFILE: {state['dataset_profile']}
    USER QUERY: {state['user_query']}
    CRITIC FEEDBACK (if revising): {state.get('critic_feedback', 'None')}
    
    Determine the best mathematical strategy. Output a structured plan."""
    
    # Use structured output to force JSON schema adherence
    structured_llm = llm.with_structured_output(AnalysisPlan)
    plan = structured_llm.invoke([SystemMessage(content=sys_prompt)])
    
    return {"analysis_plan": plan}

def statistical_node(state: AgentState):
    """Executes the statistical portion of the plan using a ReAct loop."""
    if not state["analysis_plan"].statistical_tests:
        return {"statistical_results": "No statistical tests required by plan."}

    sys_prompt = f"""You are the Statistical Agent.
    DATASET PROFILE: {state['dataset_profile']}
    TASKS TO EXECUTE: {', '.join(state['analysis_plan'].statistical_tests)}
    
    The data is at `current_data.csv`. Write Python code using `python_repl_tool` to execute these exact statistical tests. 
    Analyze the tool output. If there is an error, rewrite and fix it. 
    Conclude with the hard mathematical results (p-values, coefficients, etc)."""
    
    res = stat_agent.invoke({"messages": [SystemMessage(content=sys_prompt), HumanMessage(content="Begin statistical analysis.")]})
    return {"statistical_results": res["messages"][-1].content}

def ml_node(state: AgentState):
    """Executes the Machine Learning portion of the plan using a ReAct loop."""
    if not state["analysis_plan"].ml_tasks:
        return {"ml_results": "No ML tasks required by plan."}

    sys_prompt = f"""You are the Machine Learning Agent.
    DATASET PROFILE: {state['dataset_profile']}
    TASKS TO EXECUTE: {', '.join(state['analysis_plan'].ml_tasks)}
    
    The data is at `current_data.csv`. Write Python code using `python_repl_tool` to execute these ML tasks (e.g., PCA, Random Forest Feature Importance). 
    Analyze the tool output. Iterate if errors occur. Conclude with the final insights."""
    
    res = ml_agent.invoke({"messages": [SystemMessage(content=sys_prompt), HumanMessage(content="Begin ML analysis.")]})
    return {"ml_results": res["messages"][-1].content}

def critic_node(state: AgentState):
    """Reviews the outputs of both execution agents against the original plan."""
    sys_prompt = f"""You are the rigorous Statistical Critic.
    USER QUERY: {state['user_query']}
    PLAN: {state['analysis_plan'].json() if state['analysis_plan'] else ''}
    STATISTICAL RESULTS: {state['statistical_results']}
    ML RESULTS: {state['ml_results']}
    
    Review the actual computed artifacts. 
    Did the agents successfully execute the plan? Are the conclusions mathematically sound?
    If yes, output: "APPROVED: [Reason]"
    If no, output: "REJECTED: [Detailed instructions on what to fix]"
    """
    response = llm.invoke([SystemMessage(content=sys_prompt)])
    return {"critic_feedback": response.content, "revision_count": state["revision_count"] + 1}

def writer_node(state: AgentState):
    """Formats the final response, injecting UI chart JSON if needed."""
    sys_prompt = f"""You are the Final Writer.
    USER QUERY: {state['user_query']}
    STATISTICAL RESULTS: {state['statistical_results']}
    ML RESULTS: {state['ml_results']}
    
    Synthesize all findings into a mathematically rigorous final response. 
    If a chart is needed, use the strict JSON chart protocol:
    ```chart
    {{ "type": "line"|"bar"|"pie"|"scatter"|"area", ... }}
    ```"""
    response = llm.invoke([SystemMessage(content=sys_prompt)])
    return {"messages": [AIMessage(content=response.content)]}


# ---------------------------------------------------------
# 6. GRAPH COMPILATION
# ---------------------------------------------------------
def route_critic(state: AgentState):
    if "REJECTED" in state['critic_feedback'] and state['revision_count'] < 3:
        return "planner"
    return "writer"

workflow = StateGraph(AgentState)

workflow.add_node("profiler", profiler_node)
workflow.add_node("planner", planner_node)
workflow.add_node("statistical_agent", statistical_node)
workflow.add_node("ml_agent", ml_node)
workflow.add_node("critic", critic_node)
workflow.add_node("writer", writer_node)

workflow.add_edge(START, "profiler")
workflow.add_edge("profiler", "planner")
workflow.add_edge("planner", "statistical_agent")
workflow.add_edge("statistical_agent", "ml_agent")
workflow.add_edge("ml_agent", "critic")
workflow.add_conditional_edges("critic", route_critic, {"planner": "planner", "writer": "writer"})
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
            "analysis_plan": None,
            "statistical_results": "",
            "ml_results": "",
            "critic_feedback": "",
            "revision_count": 0
        }

        final_state = app_graph.invoke(initial_state)
        
        final_msg = final_state["messages"][-1]
        print(f"Final message content: {final_msg.content if hasattr(final_msg, 'content') else str(final_msg)}")
        response_text = final_msg.content if hasattr(final_msg, 'content') else str(final_msg)
        
        return {"response": response_text}
    
    except Exception as e:
        print(f"CRASH: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))
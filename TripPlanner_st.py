
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
import streamlit as st
from typing import List, Dict
from pydantic import BaseModel, Field
import json
from langgraph.constants import Send
from typing import Annotated
import operator

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # Enable LangSmith tracing
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

# Initialize LLM
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

# Schema for structured output from orchestrator
class TripTask(BaseModel):
    name: str = Field(description="Name of the trip planning subtask (e.g., 'Find flights').")
    description: str = Field(description="Brief description of what to do in this subtask.")

class TripPlan(BaseModel):
    tasks: List[TripTask] = Field(description="List of subtasks for planning the trip.")

# Augment LLM with structured output for planning
planner = llm.with_structured_output(TripPlan)

# Graph State
class State(TypedDict):
    trip_details: str  # User input (e.g., "5-day trip to Paris, $1000, culture")
    tasks: List[TripTask]  # List of trip planning subtasks
    worker_outputs: Annotated[List[str], operator.add]  # Collect all worker outputs as a list
    final_itinerary: str  # Final synthesized trip plan

# Worker State
class WorkerState(TypedDict):
    task: TripTask

# Nodes
def orchestrator(state: State):
    """Orchestrator that generates a plan for the trip."""
    prompt = (
        f"Analyze this trip request and identify specific subtasks for planning (e.g., 'Find flights', 'Plan activities'). "
        f"Return a JSON list of subtasks with names and descriptions. Trip details: {state['trip_details']}"
    )
    plan = planner.invoke(prompt)
    return {"tasks": plan.tasks, "worker_outputs": []}

def worker(state: WorkerState):
    """Worker processes a single trip planning subtask."""
    task_name = state["task"].name
    task_desc = state["task"].description
    prompt = (
        f"Complete this trip planning subtask: '{task_name}'. Description: '{task_desc}'. "
        f"Provide a concise response (50-500 words) with actionable details."
    )
    msg = llm.invoke(prompt)
    return {"worker_outputs": [f"{task_name}: {msg.content}"]}  # Return as a list to be aggregated

def synthesizer(state: State):
    """Synthesize worker outputs into a final trip itinerary."""
    worker_results = "\n".join(state["worker_outputs"])
    prompt = (
        f"Create a cohesive trip itinerary based on these planning subtask results. "
        f"Format as a day-by-day plan, keeping it concise (150-200 words total). Results:\n{worker_results}"
    )
    msg = llm.invoke(prompt)
    return {"final_itinerary": msg.content}

# Conditional edge function to assign workers
def assign_workers(state: State):
    """Assign a worker to each subtask in the plan."""
    return [Send("worker", {"task": t}) for t in state["tasks"]]

# Build workflow
def make_graph():
    """Create and compile the orchestrator-worker trip planning workflow."""
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("orchestrator", orchestrator)
    workflow.add_node("worker", worker)
    workflow.add_node("synthesizer", synthesizer)
    
    # Add edges
    workflow.add_edge(START, "orchestrator")
    workflow.add_conditional_edges("orchestrator", assign_workers, ["worker"])
    workflow.add_edge("worker", "synthesizer")
    workflow.add_edge("synthesizer", END)
    
    # Compile
    trip_planner_agent = workflow.compile()
    return trip_planner_agent

# Create the graph
trip_planner_agent = make_graph()

# Streamlit app
def main():
    """Run the Streamlit app for trip planning with orchestrator-worker workflow."""
    st.title("AI Trip Planner")
    st.write("Enter your trip details below (e.g., '5-day trip to Paris, $1000, culture') to get a custom itinerary!")

    # Text input for trip details
    trip_details = st.text_input("Trip Details", placeholder="e.g., 5-day trip to Paris, $1000, culture")

    # Button to generate itinerary
    if st.button("Plan My Trip"):
        if not trip_details.strip():
            st.warning("Please enter trip details!")
        else:
            with st.spinner("Planning your trip..."):
                try:
                    state = trip_planner_agent.invoke({"trip_details": trip_details})
                    st.subheader("Your Trip Itinerary")
                    st.write(state["final_itinerary"])
                except Exception as e:
                    st.error(f"Failed to plan trip: {str(e)}. Please try again later.")

if __name__ == "__main__":
    main()
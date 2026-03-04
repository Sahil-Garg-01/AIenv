from langgraph.graph import StateGraph
from typing import TypedDict
from agent.nodes import generate_report

class AgentState(TypedDict):
    hazard: str
    report: str

def build_graph():

    builder = StateGraph(AgentState)

    builder.add_node("report_node", generate_report)

    builder.set_entry_point("report_node")

    builder.set_finish_point("report_node")

    return builder.compile()
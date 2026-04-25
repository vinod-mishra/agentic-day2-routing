from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage

load_dotenv()
from typing import Literal, TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, START, END

# -------------------------------
# 1. Define State
# -------------------------------
class SupportState(TypedDict):
    # Annotated with operator.add ensures messages are appended, not overwritten
    messages: Annotated[list[BaseMessage], operator.add]
    should_escalate: bool
    issue_type: str
    user_tier: str  # "vip" or "standard"

# -------------------------------
# 2. LLM Setup
# -------------------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", seed=6)

# -------------------------------
# 3. Define Nodes
# -------------------------------

def initial_triage(state: SupportState):
    """Acknowledge request."""
    return {
        "messages": [
            AIMessage(content="Triage complete. Analyzing your request...")
        ]
    }


def classify_issue(state: SupportState):
    """Classify issue type and decide escalation."""
    last_message = state["messages"][-1].content.lower()

    if "refund" in last_message or "payment" in last_message:
        return {
            "issue_type": "billing",
            "should_escalate": True
        }

    return {
        "issue_type": "general",
        "should_escalate": False
    }


def handle_vip(state: SupportState):
    """VIP support path with LLM response."""
    response = llm.invoke(state["messages"])

    return {
        "messages": [
            AIMessage(content=f"[VIP PRIORITY] {response.content}")
        ]
    }


def handle_standard(state: SupportState):
    """Standard support path with LLM response."""
    response = llm.invoke(state["messages"])

    return {
        "messages": [
            AIMessage(content=f"[STANDARD SUPPORT] {response.content}")
        ]
    }


def escalate(state: SupportState):
    """Escalation node."""
    return {
        "messages": [
            AIMessage(content="Your issue is being escalated to a human agent.")
        ]
    }

# -------------------------------
# 4. Routing Functions
# -------------------------------
def route_by_tier(state: SupportState) -> Literal["vip_path", "standard_path"]:
    """Route based on user tier."""
    if state.get("user_tier") == "vip":
        return "vip_path"
    return "standard_path"


def route_escalation(state: SupportState) -> Literal["escalate", "end"]:
    """Route based on escalation requirement."""
    if state.get("should_escalate"):
        return "escalate"
    return "end"

# -------------------------------
# 5. Build Graph
# -------------------------------
workflow = StateGraph(SupportState)

# Nodes
workflow.add_node("triage", initial_triage)
workflow.add_node("classify", classify_issue)
workflow.add_node("vip_path", handle_vip)
workflow.add_node("standard_path", handle_standard)
workflow.add_node("escalate", escalate)

# Set Entry Point
workflow.add_edge(START, "triage")

# Sequential step
workflow.add_edge("triage", "classify")

# Tier-based routing
workflow.add_conditional_edges(
    "classify",
    route_by_tier,
    {
        "vip_path": "vip_path",
        "standard_path": "standard_path"
    }
)

# Escalation routing
workflow.add_conditional_edges(
    "vip_path",
    route_escalation,
    {
        "escalate": "escalate",
        "end": END
    }
)

workflow.add_conditional_edges(
    "standard_path",
    route_escalation,
    {
        "escalate": "escalate",
        "end": END
    }
)

# Final edge
workflow.add_edge("escalate", END)

# Compile
app = workflow.compile()


# -------------------------------
# 6. Run TEST
# -------------------------------
if __name__ == "__main__":
    initial_state = {
        "messages": [
            HumanMessage(content="I want a refund for my last payment")
        ],
        "should_escalate": False,
        "issue_type": "",
        "user_tier": "vip"  # change to "standard" to test
    }

    result = app.invoke(initial_state)

    for msg in result["messages"]:
        print(msg.content)
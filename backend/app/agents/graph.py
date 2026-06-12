from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, StateGraph

from app.agents.advisor_copilot import advisor_copilot_node
from app.agents.document_intelligence import document_intelligence_node
from app.agents.error_handler import error_handler_node
from app.agents.intake import intake_node
from app.agents.intent_classifier import intent_classifier_node
from app.agents.profile_builder import profile_builder_node
from app.agents.risk_assessment import risk_assessment_node
from app.agents.scoring import scoring_node
from app.agents.state import GraphState
from app.agents.strategy import strategy_node
from app.agents.supervisor import route_supervisor, supervisor_node
from app.guardrails.input_guard import guardrail_input_node
from app.guardrails.output_guard import guardrail_output_node


def create_graph(checkpointer: AsyncPostgresSaver):
    builder = StateGraph(GraphState)

    builder.add_node("guardrail_input", guardrail_input_node)
    builder.add_node("intent_classifier", intent_classifier_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("intake", intake_node)
    builder.add_node("document_intelligence", document_intelligence_node)
    builder.add_node("profile_builder", profile_builder_node)
    builder.add_node("risk_assessment", risk_assessment_node)
    builder.add_node("strategy", strategy_node)
    builder.add_node("scoring", scoring_node)
    builder.add_node("advisor_copilot", advisor_copilot_node)
    builder.add_node("error_handler", error_handler_node)
    builder.add_node("guardrail_output", guardrail_output_node)

    # Entry: guard → classify intent → supervisor decides
    builder.set_entry_point("guardrail_input")
    builder.add_edge("guardrail_input", "intent_classifier")
    builder.add_edge("intent_classifier", "supervisor")

    builder.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "end": END,
            "intake": "intake",
            "document_intelligence": "document_intelligence",
            "profile_builder": "profile_builder",
            "risk_assessment": "risk_assessment",
            "strategy": "strategy",
            "scoring": "scoring",
            "advisor_copilot": "advisor_copilot",
            "error_handler": "error_handler",
        },
    )

    # Pipeline nodes loop back to supervisor (not re-entering intent_classifier)
    builder.add_edge("intake", "supervisor")
    builder.add_edge("document_intelligence", "supervisor")
    builder.add_edge("profile_builder", "supervisor")
    builder.add_edge("risk_assessment", "supervisor")
    builder.add_edge("strategy", "supervisor")
    builder.add_edge("scoring", "supervisor")

    # advisor_copilot → output guardrail (adds disclaimer) → END
    builder.add_edge("advisor_copilot", "guardrail_output")
    builder.add_edge("guardrail_output", END)
    builder.add_edge("error_handler", END)

    return builder.compile(checkpointer=checkpointer)

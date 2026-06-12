from datetime import datetime, timezone
from io import BytesIO

from bson import ObjectId
from langchain_core.messages import AIMessage
from pypdf import PdfReader

from app.agents.state import GraphState
from app.db.mongo import get_mongo_db
from app.llm.client import get_chat_model
from app.observability.langfuse_setup import traced_node

_MAX_CHARS = 12_000

_EXTRACT_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_document_data",
        "description": "Extract investment holdings and account info from a brokerage statement.",
        "parameters": {
            "type": "object",
            "properties": {
                "holdings": {
                    "type": "array",
                    "description": "All investment positions found in the document.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string", "description": "Ticker symbol e.g. 'AAPL'"},
                            "shares": {"type": "number", "description": "Number of shares held"},
                            "value": {"type": "number", "description": "Current market value in USD"},
                        },
                        "required": ["ticker", "value"],
                    },
                },
                "account_value": {
                    "type": "number",
                    "description": "Total account value in USD",
                },
                "account_type": {
                    "type": "string",
                    "description": "Account type: brokerage, IRA, 401k, Roth IRA, etc.",
                },
                "institution": {
                    "type": "string",
                    "description": "Financial institution name e.g. 'Fidelity', 'Vanguard'",
                },
            },
        },
    },
}

_SYSTEM_PROMPT = (
    "You are a financial document parser. Extract all investment holdings and account summary "
    "data from the brokerage statement text provided. Call extract_document_data with every "
    "position you can identify. If a field is not present, omit it from the call."
)


def _pdf_to_text(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
            if len(text) >= _MAX_CHARS:
                break
        return text[:_MAX_CHARS]
    except Exception:
        return ""


@traced_node("document_intelligence")
async def document_intelligence_node(state: GraphState) -> dict:
    mongo = get_mongo_db()
    doc_ids = state.get("documents_uploaded") or []

    # Find which docs have already been extracted
    extracted_ids: set[str] = set()
    async for rec in mongo["extracted_document_data"].find(
        {"session_id": state["session_id"]}, {"doc_id": 1}
    ):
        extracted_ids.add(rec["doc_id"])

    llm = get_chat_model().bind_tools([_EXTRACT_TOOL])

    for doc_id_str in doc_ids:
        if doc_id_str in extracted_ids:
            continue

        raw_doc = await mongo["raw_documents"].find_one({"_id": ObjectId(doc_id_str)})
        if not raw_doc:
            continue

        pdf_text = _pdf_to_text(raw_doc["content"])

        extraction: dict = {}
        if pdf_text.strip():
            response = await llm.ainvoke([
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Document text:\n\n{pdf_text}"},
            ])
            if response.tool_calls:
                extraction = response.tool_calls[0]["args"]

        await mongo["extracted_document_data"].insert_one({
            "session_id": state["session_id"],
            "doc_id": doc_id_str,
            "extraction": extraction,
            "extracted_at": datetime.now(timezone.utc),
        })

    return {
        "documents_extracted": True,
        "messages": [AIMessage(
            content="Your documents have been processed. I'll now build your profile.",
            name="document_intelligence",
        )],
    }


import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent.agent import run_investigator
from middlewares.exceptions import SemanticValidationError, register_exception_handlers
from middlewares.middleware import register_middleware
from middlewares.schemas import AnalyzeTicketRequest, AnalyzeTicketResponse


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_middleware(app)
register_exception_handlers(app)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/analyze-ticket", response_model=AnalyzeTicketResponse)
def analyze_ticket(payload: AnalyzeTicketRequest):
    """Analyze a customer support ticket.

    Forwards the validated request to the multi-agent investigator
    (``agent.agent.run_investigator``) and returns its structured response.
    All three guardrail layers (prompt-injection ``before_agent``,
    LLM-as-judge ``after_agent``, deterministic orchestrator scrub) run
    inside the agent pipeline; the catch-all exception handler in
    ``middlewares.exceptions`` ensures a non-sensitive 500 is returned if
    the pipeline fails for any reason (Section 9.2).
    """
    # Defensive semantic check: pydantic's min_length=1 already catches an
    # empty complaint, but a whitespace-only complaint would otherwise slip
    # through. Surface as 422 (semantic) rather than letting the agent
    # process garbage text.
    if not payload.complaint.strip():
        raise SemanticValidationError("complaint must not be empty or whitespace only")

    return run_investigator(payload)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8002)),
        reload=True,
    )